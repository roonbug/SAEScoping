from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import tqdm
from beartype import beartype
from beartype.typing import Any, Callable, Literal
from datasets import Dataset

# https://docs.kidger.site/jaxtyping/api/array/
from jaxtyping import Float, Integer, jaxtyped
from sae_lens import SAE, JumpReLUSAE
from transformers import (
    Gemma2ForCausalLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Olmo2ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks


class _Gemma2SFTTrainer(SFTTrainer):
    """Workaround for TRL 0.22.x + multi-GPU Gemma-2 bug: the Trainer sends a batch
    sized (per_device_batch * n_gpu) but the model on a single device returns logits
    sized (per_device_batch * n_gpu, seq, vocab). TRL's entropy computation then fails
    because per_token_entropy shape doesn't match attention_mask shape when the model
    is placed on a single device via device_map while n_gpu > 1.

    We override compute_loss to make the entropy computation use the logits-matching
    attention_mask shape."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_eval_dataset_name: str | None = None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # When Trainer iterates over a dict of eval datasets, it calls
        # evaluate(metric_key_prefix="eval_<name>"). Extract the dataset name
        # so compute_loss can store metrics under dataset-specific keys.
        if metric_key_prefix.startswith("eval_"):
            self._current_eval_dataset_name = metric_key_prefix[len("eval_"):]
        else:
            self._current_eval_dataset_name = None
        result = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        self._current_eval_dataset_name = None
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        inputs["use_cache"] = False

        # Save attention_mask before parent modifies anything
        saved_mask = inputs["attention_mask"].clone() if "attention_mask" in inputs else None

        (loss, outputs) = super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute entropy with shape-safe mask
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                # Chunk over sequence dim to avoid OOM on large vocab (256k for Gemma-2).
                _entropy_chunks = []
                for _c in outputs.logits.split(128, dim=1):
                    _lp = torch.nn.functional.log_softmax(_c, dim=-1)
                    _entropy_chunks.append(-(torch.exp(_lp) * _lp).sum(-1))
                per_token_entropy = torch.cat(_entropy_chunks, dim=1)
                if saved_mask is not None:
                    attention_mask = saved_mask
                    # Align batch dimensions if they differ (multi-GPU dataloader + single-GPU model)
                    if per_token_entropy.shape[0] != attention_mask.shape[0]:
                        attention_mask = attention_mask[: per_token_entropy.shape[0]]
                    virtual_attention_mask = torch.ones(
                        attention_mask.size(0), self.num_virtual_tokens, device=attention_mask.device
                    )
                    attention_mask = torch.cat((virtual_attention_mask, attention_mask), dim=1)
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                del per_token_entropy
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            if mode == "eval" and self._current_eval_dataset_name:
                entropy_key = f"{self._current_eval_dataset_name}_entropy"
            else:
                entropy_key = "entropy"
            self._metrics[mode][entropy_key].append(entropy)

        if mode == "train":
            if saved_mask is not None:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(saved_mask.sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        if mode == "eval" and self._current_eval_dataset_name:
            num_tokens_key = f"{self._current_eval_dataset_name}_num_tokens"
        else:
            num_tokens_key = "num_tokens"
        self._metrics[mode][num_tokens_key] = [self._total_train_tokens]

        # Compute token accuracy
        if "labels" in inputs and not self.args.use_liger_kernel:
            with torch.no_grad():
                # Compute argmax in chunks to avoid OOM from materializing the full logits slice
                logits = outputs.logits[..., :-1, :]  # (B, T-1, V) — no contiguous yet
                shift_labels = inputs["labels"][..., 1:].contiguous()
                if logits.shape[0] != shift_labels.shape[0]:
                    shift_labels = shift_labels[: logits.shape[0]]
                preds = logits.argmax(dim=-1)  # (B, T-1) — much smaller
                del logits
                valid = shift_labels != -100
                correct = (preds == shift_labels) & valid
                accuracy = correct.sum().float() / valid.sum().float() if valid.sum() > 0 else torch.tensor(0.0)
                accuracy = self.accelerator.gather_for_metrics(accuracy).mean().item()
            if mode == "eval" and self._current_eval_dataset_name:
                acc_key = f"{self._current_eval_dataset_name}_mean_token_accuracy"
            else:
                acc_key = "mean_token_accuracy"
            self._metrics[mode][acc_key].append(accuracy)

        if not return_outputs:
            return loss
        return loss, outputs

# Our libraries
from sae_scoping.utils.hooks.sae import (
    SAEWrapper,
    Context,
    SAELensEncDecCallbackWrapper,
)
from sae_scoping.trainers.sae_enhanced.utils import str_dict_diff

"""
Train a model with SFT while under hooks. Limit the set of modified parameters to
those after the SAE.
"""


@beartype
def _freeze_layers(
    model: PreTrainedModel, layers_to_freeze: list[int]
) -> list[str]:
    frozen_set = set(layers_to_freeze)
    parameters_to_freeze = []
    if type(model) not in [
        Gemma2ForCausalLM,
        LlamaForCausalLM,
        AutoModelForCausalLM,
        Gemma3ForConditionalGeneration,
        Olmo2ForCausalLM,
    ]:
        raise ValueError(f"Model {type(model)} is not supported")
    # Use the correct layer prefix for each model family.
    # Gemma3 nests layers under model.language_model; Gemma2/Llama use model.layers.
    layer_prefix = "model.language_model.layers" if model.config.model_type in {"gemma3"} else "model.layers"
    layer_patt = rf"^{re.escape(layer_prefix)}\.(\d+)\..*$"
    for n, p in model.named_parameters():
        if not n.startswith(layer_prefix):
            if "lm_head" in n:
                p.requires_grad = True
            elif type(model) == Gemma2ForCausalLM and n.startswith("model.norm"):
                p.requires_grad = True
            else:
                # Freeze all non-layer parameters (embeddings, etc.)
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
                parameters_to_freeze.append(n)
        else:
            match = re.match(layer_patt, n)
            assert match is not None, (
                f"Parameter name {n} doesn't match expected pattern"
            )
            layer_num = int(match.group(1))
            if layer_num in frozen_set:
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
                parameters_to_freeze.append(n)
            else:
                p.requires_grad = True
    return parameters_to_freeze


def train_sae_enhanced_model(
    train_dataset: Dataset,
    eval_dataset: Dataset | dict[str, Dataset],  # to eval on multiple OOD datasets
    sae,  # SAE | SAELensEncDecCallbackWrapper | nn.Module | None
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    T: float | int = 0.0,
    hookpoint: str | None = "",
    save_output: bool = False,
    return_trained_model: bool = False,
    all_layers_after_hookpoint: bool = False,
    # TODO(Adriano) add support for better callbacks
    training_callbacks: list[TrainerCallback] = [],
    sft_config: SFTConfig | None = None,  # None => use default (one below)
    **kwargs: dict[str, Any],
) -> PreTrainedModel | None:
    wandb_project_name = kwargs.get(
        "wandb_project_name", os.environ.get("WANDB_PROJECT", None)
    )
    if wandb_project_name is None:
        raise ValueError("WANDB_PROJECT is not set")
    wandb_run_name = kwargs.get(
        "wandb_run_name", os.environ.get("WANDB_RUN_NAME", None)
    )
    resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
    old_environ_name = os.environ.get("WANDB_PROJECT", None)
    try:
        # 1. setup SFT arguments
        os.environ["WANDB_PROJECT"] = wandb_project_name
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
        default_sft_config = SFTConfig(
            run_name=wandb_run_name,  # None => use default
            output_dir=kwargs.get("output_dir", "./deleteme_sft_output"),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.1,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            save_steps=kwargs.get("save_steps", 1_000_000),  # Do not intend to save
            save_strategy="no",  # Do not intend to save
            logging_steps=10,
            fp16=False,
            bf16=True,  # H100/A100 hopefully will work here? Llama2
            remove_unused_columns=False,
            eval_strategy="steps",
            eval_steps=100,  # wanna do this somewhat often, but not tooo much
            save_total_limit=2,
            # load_best_model_at_end=True, # <- can't do this w/out matching save/eval strat
            # metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            max_steps=kwargs.get("max_steps", 1_000),
            max_length=kwargs.get("context_length", 1024),
            # TODO(adriano) don't hardcode
            gradient_checkpointing=False,
        )
        if sft_config is None:
            sft_config = default_sft_config

        # 2. freeze (and sanity check/prinout)
        # NOTE: even if no SAE, then you COULD still pass in a hookpoint to limit which
        # layers are trained
        if sae is not None and hookpoint is None:
            raise ValueError(
                "If SAE is provided, then you must also provide a hookpoint"
            )
        p2f = set()
        if hookpoint is not None:
            hp_patt = r"^model\.language_model\.layers\.(\d+)$" if model.config.model_type in {"gemma3"} else r"^model\.layers\.(\d+)$"
            if not re.match(hp_patt, hookpoint):
                raise ValueError(
                    f"Hookpoint {hookpoint} is not a valid layer hookpoint"
                )
            sae_layer = int(re.match(hp_patt, hookpoint).group(1))
            if all_layers_after_hookpoint:
                frozen_layers = list(range(sae_layer + 1))
            else:
                frozen_layers = list(range(sae_layer + 1)) + list(range(sae_layer + 2, len(model.language_model.layers) - 1)) if model.config.model_type in {"gemma3"} else list(range(sae_layer + 1)) + list(range(sae_layer + 2, len(model.model.layers) - 1))
            p2f = set(_freeze_layers(model, frozen_layers))
        trainable_params_be4 = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_be4 = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        _frozen_layers_str = f", frozen layers={frozen_layers}" if hookpoint is not None else ""
        print(
            f"Params @ hookpoint={hookpoint}: "
            f"{len(trainable_params_be4)} trainable, {len(frozen_params_be4)} frozen"
            f"{_frozen_layers_str}"
        )
        # copy a small word; surely the words will change w.h.p. or smth?
        p2s1 = {
            n: p.data.detach().view(-1)[:32].cpu() for n, p in model.named_parameters()
        }

        # 3. Setup and train
        trainer = _Gemma2SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=training_callbacks,
        )
        trainable_params_after = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_after = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        assert trainable_params_be4 == trainable_params_after
        assert frozen_params_be4 == frozen_params_after
        if isinstance(resume_from_checkpoint, str):
            opt_path = Path(resume_from_checkpoint) / "optimizer.pt"
            if opt_path.exists():
                print(f"Loading optimizer and scheduler states from {resume_from_checkpoint}")
            else:
                print(f"WARNING: resume_from_checkpoint={resume_from_checkpoint} but no optimizer.pt found — optimizer starts fresh")

        if sae is not None:
            # This will work no matter which of the above types you use
            sae_wrapper = SAEWrapper(sae)
            hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
            with named_forward_hooks(model, hook_dict):
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Sanity
        trainable_params_end = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_end = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        assert trainable_params_be4 == trainable_params_end
        assert frozen_params_be4 == frozen_params_end

        # SAnity check we learend but not on the forzen ones
        parameters_that_changed = []
        for n, p in model.named_parameters():
            slc = p.data.detach().view(-1)[:32].cpu()
            if not torch.allclose(slc, p2s1[n]):
                parameters_that_changed.append(n)
        # fmt: off
        assert len(set(parameters_that_changed) & set(frozen_params_end)) == 0, f"Parameters that changed and are frozen: {json.dumps(list(parameters_that_changed & set(frozen_params_end)), indent=4)}\n\nShould be empty"
        assert len(set(parameters_that_changed) & p2f) == 0, f"Parameters that changed and are frozen: {json.dumps(list(parameters_that_changed & p2f), indent=4)}\n\nShould be empty"
        # fmt: on

        if save_output:
            trainer.save_model()
        if return_trained_model:
            return model

    finally:
        if old_environ_name is not None:
            os.environ["WANDB_PROJECT"] = old_environ_name


if __name__ == "__main__":
    from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
    from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae

    def test_end2end():
        # Try a simple integration test with a dummy model and gemmascope
        print("=" * 100)
        print("Importing dependencies")
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("=" * 100)
        print("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        tokenizer_chat = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        assert tokenizer_chat.chat_template is not None

        print("=" * 100)
        print("Loading dataset")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=1)
        n_samples_ranking = 100
        n_samples_training = 200
        n_samples_evaluation = 50
        batch_size = 128
        n_samples_total = n_samples_ranking + n_samples_training + n_samples_evaluation
        assert len(dataset) >= n_samples_total
        dataset = dataset.select(range(n_samples_total))
        dataset = dataset.map(
            lambda x: {
                "text": tokenizer_chat.apply_chat_template(
                    [
                        {"role": "user", "content": x["question"]},
                        {"role": "assistant", "content": x["answer"]},
                    ],
                    tokenize=False,
                )
            },
            batched=False,  # It's pretty fast so should be fine tbh
        )
        dataset_ranking = dataset.select(range(n_samples_ranking))
        dataset_training = dataset.select(
            range(n_samples_ranking, n_samples_ranking + n_samples_training)
        )
        dataset_evaluation = dataset.select(
            range(n_samples_ranking + n_samples_training, n_samples_total)
        )

        print("=" * 100)
        print("Loading SAE and moving all models to device")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = model.to(device)
        sae = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_0/width_16k/canonical",
            device=device,
        )
        sae = sae.to(device)  # defensive likely-noop
        print("Sanity check types:")
        print(type(sae))
        print(type(sae).__mro__)
        assert isinstance(sae, JumpReLUSAE)  # ^

        print("=" * 100)
        print("Creating hookpoint and ranking neurons")
        # hookpoint = "blocks.0.hook_resid_post" # this is the HookedTransformer hookpoint
        hookpoint = "model.language_model.layers.0"  # register as post-hook; default

        T = 0
        p = 0.5  # Keep the top 50% of neurons
        batch_size = 32
        ranking, distribution = rank_neurons(
            dataset=dataset_ranking,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
            hookpoint=hookpoint,
            batch_size=batch_size,
            return_distribution=True,
        )

        @jaxtyped(typechecker=beartype)
        def plot_distribution(
            distribution: Float[torch.Tensor, "d_sae"],
            file_path: Path | None = None,
        ) -> None:
            distribution_np = distribution.detach().cpu().numpy()
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(distribution_np)), distribution_np)
            plt.xlabel("Neuron index")
            plt.ylabel("Firing count")
            plt.title("Firing count distribution")
            if file_path is not None:
                plt.savefig(file_path)
            else:
                plt.show()

        save_distribution_unsorted_path = (
            Path(__file__).parent / "deleteme_fc_unsorted.png"
        )
        save_distribution_sorted_path = Path(__file__).parent / "deleteme_fc_sorted.png"
        plot_distribution(distribution, save_distribution_unsorted_path)
        plot_distribution(distribution[ranking], save_distribution_sorted_path)

        print("=" * 100)
        print("Getting pruned SAE (just wrapper tbh)")
        pruned_sae: SAELensEncDecCallbackWrapper = get_pruned_sae(sae, ranking, p, T)
        print(pruned_sae)

        print("=" * 100)
        print("Running inference with pruned SAE (to make sure it works OK)")
        sw = SAEWrapper(pruned_sae)
        fn = partial(filter_hook_fn, sw)
        hook_dict = {hookpoint: fn}
        with torch.no_grad():
            for i in tqdm.trange(0, len(dataset_evaluation), batch_size):
                texts = dataset_evaluation["text"][
                    i : min(i + batch_size, len(dataset_evaluation))
                ]
                assert all(isinstance(text, str) for text in texts)
                batch = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                batch["labels"] = batch["input_ids"]  # For loss calculation; hf shifts
                with named_forward_hooks(model, hook_dict):
                    loss_with_sae = model(**batch).loss
                loss_without_sae = model(**batch).loss
                info_printout = {
                    "Loss w/ SAE": loss_with_sae.item(),
                    "Loss w/o SAE": loss_without_sae.item(),
                }
                print(json.dumps(info_printout, indent=4))

        print("=" * 100)
        print("Training SAE-enhanced model")
        train_sae_enhanced_model(
            train_dataset=dataset_training,
            eval_dataset=dataset_evaluation,
            sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
            hookpoint=hookpoint,
            save_output=False,
            training_callbacks=[],
            sft_config=None,  # Use default one
            # These get populated into the SFT config
            wandb_project_name="deleteme_gemma_sae_recovery_training",
            wandb_run_name="debugging/layer_0--width_16k--canonical",
        )

    test_end2end()
