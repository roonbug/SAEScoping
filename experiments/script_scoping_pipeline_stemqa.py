"""
Full SAE scoping pipeline. Supports Gemma-2-9b-it (--gemma2) and Gemma-3-12b-it (--gemma3, default).
# Working transformers version 4.56.1
Stages:
  1. RANK:     Compute firing rates on the chosen train domain (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < threshold
  3. RECOVER:  In-domain SFT on the train domain
  4. ATTACK:   Adversarial SFT on an OOD domain

Supported train domains: biology, chemistry, math, physics
The remaining domains are automatically used for OOD eval.

Usage:
  # Full pipeline, biology domain, gemma3 (default):
  python script_scoping_pipeline_stemqa.py --train-domain biology --attack-domain chemistry --stage all

  # Same but with gemma2-9b:
  python script_scoping_pipeline_stemqa.py --train-domain biology --attack-domain chemistry --stage all --gemma2

  # Just recovery training (assumes firing rates already computed):
  python script_scoping_pipeline_stemqa.py --train-domain biology --stage recover

  # Just adversarial training (assumes recovery checkpoint exists):
  python script_scoping_pipeline_stemqa.py --train-domain biology --stage attack \
      --attack-domain chemistry --checkpoint outputs_scoping/biology/recover/checkpoint-XXXX
"""

from __future__ import annotations

import gc
import io
import json
import random
import re
import shutil
from pathlib import Path
import time

import click
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi, snapshot_download
from itertools import islice
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from trl import SFTConfig
import tqdm
import sys
import os
sys.path.append(os.path.abspath(".."))
from functools import partial
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.xxx_evaluation.trainer_callbacks import LLMJudgeScopingTrainerCallback
from script_scoping_pipeline_stemqa_learnsae import (
    _load_sae_from_cache,
    _ae_filter,
    stage_train as _sparse_stage_train,
)
from sparsify import SparseCoder


class _HfCheckpointCallback(TrainerCallback):
    """Captures the wandb run ID and uploads each checkpoint to HF immediately after it is saved."""

    def __init__(self):
        self.run_id: str | None = None
        self._api: HfApi | None = None
        self.failed_checkpoints: list[Path] = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import wandb
        if wandb.run is not None:
            if self._api is None:
                self._api = HfApi()
            bare_id = wandb.run.id
            repo_url = self._api.create_repo(repo_id=bare_id, exist_ok=True, repo_type="model")
            self.run_id = repo_url.repo_id  # e.g. "arunasank/gcjg134f"

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.run_id is None:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        if self._api is None:
            self._api = HfApi()
        print(f"Uploading {ckpt_dir.name} to HuggingFace Hub {self.run_id!r}...")
        try:
            self._api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=self.run_id,
                path_in_repo=ckpt_dir.name,
                repo_type="model",
            )
            print(f"Upload successful, deleting local {ckpt_dir.name}...")
            shutil.rmtree(ckpt_dir)
        except Exception as e:
            print(f"Warning: failed to upload {ckpt_dir.name} ({e}); will retry at end of stage.")
            self.failed_checkpoints.append(ckpt_dir)

    def retry_failed_checkpoints(self) -> None:
        """Retry checkpoints that failed during training. Exits on persistent failure."""
        if not self.failed_checkpoints or self.run_id is None:
            return
        if self._api is None:
            self._api = HfApi()
        still_failed = []
        for ckpt_dir in self.failed_checkpoints:
            if not ckpt_dir.exists():
                continue
            print(f"Retrying upload of {ckpt_dir.name} to HuggingFace Hub {self.run_id!r}...")
            try:
                self._api.upload_folder(
                    folder_path=str(ckpt_dir),
                    repo_id=self.run_id,
                    path_in_repo=ckpt_dir.name,
                    repo_type="model",
                )
                print(f"Retry successful, deleting local {ckpt_dir.name}...")
                shutil.rmtree(ckpt_dir)
            except Exception as e:
                still_failed.append(ckpt_dir)
                print(f"Retry failed for {ckpt_dir.name}: {e}")
        if still_failed:
            sys.exit(f"ERROR: {len(still_failed)} checkpoint(s) could not be uploaded after retry: {still_failed}")

# ── Model configs ─────────────────────────────────────────────────────────────
GEMMA3_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_31_width_16k_l0_medium",
    hookpoint="model.language_model.layers.31",
    cache_tag="layer_31--width_16k--canonical",
)

# GEMMA3_CONFIG = dict(
#     model_name="google/gemma-3-12b-it",
#     sae_release="gemma-scope-2-12b-it-res-all",
#     sae_id="layer_39_width_262k_l0_small",
#     hookpoint="model.language_model.layers.39",
#     cache_tag="layer_39--width_262k--canonical",
# )
GEMMA2_CONFIG = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_31/width_16k/canonical",
    hookpoint="model.layers.31",
    cache_tag="layer_31--width_16k--canonical",
)
GEMMA3_LATER_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_41_width_16k_l0_medium",
    hookpoint="model.language_model.layers.41",
    cache_tag="layer_41--width_16k--canonical",
)

GEMMA3_CONFIG_131K = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_31_width_16k_l0_small",
    hookpoint="model.language_model.layers.31",
    cache_tag="layer_31--width_16k--small",
)
GEMMA2_CONFIG_131K = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_31/width_131k/canonical",
    hookpoint="model.layers.31",
    cache_tag="layer_31--width_131k--canonical",
)
GEMMA3_LATER_CONFIG_131K = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_41_width_16k_l0_small",
    hookpoint="model.language_model.layers.41",
    cache_tag="layer_41--width_16k--small",
)
# OLMo has no Gemmascope SAE; rank/prune stages are unavailable.
# Use with --domain-sae-path and a learned SAE from script_scoping_pipeline_stemqa_learnsae.py.
OLMO_CONFIG = dict(
    model_name="allenai/OLMo-2-1124-7B-Instruct",
    sae_release=None,
    sae_id=None,
    hookpoint="model.layers.24",
    cache_tag="layer_24",
)
FIRING_RATE_THRESHOLD = 1e-4  # 0.0001

ALL_DOMAINS = ["biology", "chemistry", "math", "physics", "coding"]
ATTACK_DOMAINS = ALL_DOMAINS

# StemQA domains share the same HF dataset.
STEMQA_DOMAINS = {"biology", "chemistry", "math", "physics"}


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _stream_qa_dataset(
    dataset_name: str,
    config: str,
    split: str,
    n_samples: int,
    tokenizer: PreTrainedTokenizerBase,
    question_col: str = "question",
    answer_col: str = "answer",
    seed: int = 1,
    stream_flag: bool = True,
) -> Dataset:
    """Stream n_samples from a HF dataset, apply chat template, return Dataset with 'text' column."""
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples... with streaming={stream_flag}")
    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag:
        stream = stream.shuffle(seed=seed, buffer_size=50)
    rows = []
    print(f"Processing stream and applying chat template... ", n_samples)
    for i, example in tqdm.tqdm(enumerate(stream), total=n_samples):
        if i >= n_samples:
            break
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example[question_col])},
                {"role": "assistant", "content": str(example[answer_col])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text, "question": str(example[question_col]), "answer": str(example[answer_col])})
    return Dataset.from_list(rows)



def load_coding_train_eval(
    tokenizer: PreTrainedTokenizerBase,
    eval_fraction: float = 0.2,
    seed: int = 42,
    n_samples: int = 60_000,
) -> tuple[Dataset, Dataset]:
    """Load nvidia/OpenCodeReasoning (non-HARD), apply chat template, return train/eval split."""
    stream = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=1000)
    rows = []
    for ex in tqdm.tqdm(stream, desc="Loading OpenCodeReasoning"):
        if "HARD" in str(ex.get("difficulty", "")).upper():
            continue
        q = ex.get("input")
        if not q or q == "-":
            continue
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": ex.get("output", "")}],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text, "question": q, "answer": ex.get("output", "")})
        if len(rows) >= n_samples:
            break
    full = Dataset.from_list(rows).shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    return full.select(range(n_eval, len(full))), full.select(range(n_eval))


def load_domain_train_eval(
    domain: str,
    tokenizer: PreTrainedTokenizerBase,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load a domain dataset and split into non-overlapping 80/20 train/eval subsets."""
    if domain == "coding":
        return load_coding_train_eval(tokenizer, eval_fraction, seed)
    elif domain in STEMQA_DOMAINS:
        full = _stream_qa_dataset(
            "4gate/StemQAMixture", domain, "train", 50_000, tokenizer, stream_flag=False
        )
    else:
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {ATTACK_DOMAINS}")

    full = full.shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  {domain} split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap, {eval_fraction:.0%} eval)")
    return train_ds, eval_ds



# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_rank(
    train_dataset: Dataset,
    n_samples: int,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    device: torch.device,
    cache_dir: Path,
    sae_release: str,
    sae_id: str,
    hookpoint: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Compute firing rates on the train domain split."""
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    n_samples = min(n_samples, len(train_dataset))
    print(f"Computing firing rates on {n_samples} train samples...")
    dataset = train_dataset.select(range(n_samples))

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device='cpu')
    sae = sae.to(device)

    ranking, distribution = rank_neurons(
        dataset=dataset,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        T=0,
        hookpoint=hookpoint,
        batch_size=batch_size,
        token_selection="attention_mask",
        return_distribution=True,
    )

    ranking = ranking.detach().cpu()
    distribution = distribution.detach().cpu()

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"ranking": ranking, "distribution": distribution}, str(cache_path))
    print(f"Saved firing rates to {cache_path}")

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    return ranking, distribution


def stage_prune(
    distribution: torch.Tensor,
    ranking: torch.Tensor,
    device: torch.device,
    sae_release: str,
    sae_id: str,
    firing_rate_threshold: float = FIRING_RATE_THRESHOLD,
):
    """Stage 2: Prune SAE at threshold, return pruned SAE wrapper."""
    n_kept = int((distribution >= firing_rate_threshold).sum().item())
    d_sae = len(distribution)
    print(f"Pruning: keeping {n_kept}/{d_sae} neurons (threshold={firing_rate_threshold})")

    # Re-sort by distribution since rank_neurons returns argsort of counts
    neuron_ranking = torch.argsort(distribution, descending=True)

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device='cpu')
    sae = sae.to(device)

    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)

    return pruned_sae, sae, n_kept


def stage_train(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    pruned_sae,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    hookpoint: str,
    output_dir: str,
    wandb_project: str,
    wandb_run: str,
    max_steps: int,
    batch_size: int,
    accum: int,
    save_every: int,
    training_callbacks=None,
    all_layers_after_hookpoint: bool = False,
    resume_from_checkpoint: bool | str = True,
    max_seq_length: int = 1024,
):
    """Stage 3/4: SFT with pruned SAE (or SparseCoder) in the loop."""
    if isinstance(pruned_sae, SparseCoder):
        _sparse_stage_train(
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            ae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=output_dir,
            wandb_project=wandb_project,
            wandb_run=wandb_run,
            max_steps=max_steps,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=training_callbacks,
            all_layers_after_hookpoint=all_layers_after_hookpoint,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        return

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
        packing=False,
        gradient_accumulation_steps=accum,
        eval_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=save_every,
        bf16=True,
        save_total_limit=5,
        report_to="wandb",
        max_length=max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
    )

    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint,
        all_layers_after_hookpoint=all_layers_after_hookpoint,
        sft_config=sft_config,
        wandb_project_name=wandb_project,
        wandb_run_name=wandb_run,
        training_callbacks=training_callbacks or [],
        resume_from_checkpoint=resume_from_checkpoint,
    )


# ── Baseline eval ─────────────────────────────────────────────────────────────

def run_baseline_eval(
    model,
    tokenizer,
    domain_questions: dict[str, list[str]],
    train_domain: str,
    wandb_project: str,
    wandb_run: str,
    csv_path: Path,
    metric_prefix: str,
    n_max_openai_requests: int = 1_800,
    attack_domain: str | None = None,
    pruned_sae=None,
    hookpoint: str | None = None,
    chart_suffix: str | None = None,
    domain_answers: dict[str, list[str]] | None = None,
    domain_generation_kwargs: dict[str, dict] | None = None,
) -> None:
    """Run LLM judge eval before training, save CSV, and log to W&B.

    If pruned_sae and hookpoint are provided, inference runs with the SAE hooked
    in so scores reflect the pruned model rather than the raw model.
    """
    evaluator = OneClickLLMJudgeScopingEval(
        n_max_openai_requests=200_000,
        train_domain=train_domain,
        attack_domain=attack_domain,
        domain_generation_kwargs=domain_generation_kwargs or {},
    )

    _cache_valid = False
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Only check the questions that would actually be sampled (deterministic seed=42).
        _sampled_questions = {
            q
            for qs in domain_questions.values()
            for q in random.Random(42).sample(qs, min(evaluator.n_samples, len(qs)))
        }
        _cache_valid = _sampled_questions.issubset(set(df["seed"]))
        if _cache_valid:
            print(f"Loading cached baseline eval from {csv_path}")
            scores = evaluator._extract_scores(df, domain_questions)
            scores_path = csv_path.with_suffix(".scores.json")
            scores_path.write_text(json.dumps(scores, indent=2))
            print(f"Saved to {csv_path} and {scores_path}")
        else:
            print(f"Cached CSV at {csv_path} is missing domains, regenerating...")
    if not _cache_valid:
        print(f"\n{'='*80}\nBaseline LLM judge eval ({wandb_run})\n{'='*80}")
        if pruned_sae is not None:
            assert hookpoint is not None, "hookpoint required when pruned_sae is provided"
            print(f"  (running with pruned SAE hooked at {hookpoint})")
        if pruned_sae is None:
            hook_dict = {}
        elif isinstance(pruned_sae, SparseCoder):
            hook_dict = {hookpoint: partial(filter_hook_fn, partial(_ae_filter, pruned_sae))}
        else:
            hook_dict = {hookpoint: partial(filter_hook_fn, SAEWrapper(pruned_sae))}
        evaluator.judge_inputs_save_dir = csv_path.parent
        with torch.no_grad(), named_forward_hooks(model, hook_dict):
            scores, df_as_json = evaluator.evaluate(
                model, tokenizer, domain_questions,
                n_max_openai_requests=n_max_openai_requests,
                domain_answers=domain_answers,
            )
        print("@" * 80)
        print("Baseline scores:")
        for k, v in sorted(scores.items()):
            print(f"  {k}: {v:.4f}")
        print("@" * 80)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        df.to_csv(csv_path, index=False)
        scores_path = csv_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved to {csv_path} and {scores_path}")

    if wandb.run is None:
        wandb.init(project=wandb_project, name=wandb_run, resume="allow", settings=wandb.Settings(init_timeout=180))
    wandb.log({f"{metric_prefix}/{k}": v for k, v in scores.items()} | {"trainer/global_step": 0})
    if chart_suffix is not None:
        wandb.log({f"{k}_{chart_suffix}": v for k, v in scores.items()} | {"trainer/global_step": 0})


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--train-domain",
    type=click.Choice(ALL_DOMAINS),
    default="biology",
    show_default=True,
    help="Domain used for SAE filtering (ranking) and recovery training.",
)
@click.option(
    "--attack-domain",
    type=click.Choice(ATTACK_DOMAINS),
    default=None,
    help="Domain used for attack training. Required when --stage is 'attack' or 'all'.",
)
@click.option(
    "--stage",
    type=str,
    default="all",
    help="Pipeline stage(s) to run: all, rank, recover, attack, or comma-separated like 'rank,recover' or 'recover,attack'.",
)
@click.option("--n-rank-samples", type=int, default=10_000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-recover", type=int, default=3_000)
@click.option("--max-steps-attack", type=int, default=4_000)
@click.option("--save-every", type=int, default=500)
@click.option("--firing-rate-threshold", type=float, default=FIRING_RATE_THRESHOLD)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Base output directory (default: experiments/outputs_scoping/<domain>)",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Recovery checkpoint path (for attack stage)",
)
@click.option(
    "--hf-recover-repo",
    type=str,
    default=None,
    help="HuggingFace repo ID of the recover model to pull (for standalone --stage attack)",
)
@click.option(
    "--hf-attack-repo",
    type=str,
    default=None,
    help="HuggingFace repo ID of a previous attack run (used with --checkpoint N to resume from step N)",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False, help="Use gemma-2-9b-it + gemma-scope-9b-pt-res SAE")
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False, help="Use gemma-3-12b-it + gemma-scope-2-12b-it-res SAE (default)")
@click.option("--gemma3-later", "later_gemma3", is_flag=True, default=False, help="Use later gemma-3-12b-it + gemma-scope-2-12b-it-res SAE")
@click.option("--olmo", "use_olmo", is_flag=True, default=False, help="Use OLMo-2-7B-Instruct (no Gemmascope SAE; requires --domain-sae-path)")
@click.option("--dev/--prod", "dev", default=True, help="Dev mode (default): cap eval at 500 samples each; prod mode: use full 20%% eval split")
@click.option("--all-layers-recover", "all_layers_recover", is_flag=True, default=False, help="Train all layers after hookpoint during recovery (default: only layer+1 and last)")
@click.option("--131k", "_131k", is_flag=True, default=False, help="Use the 131k-width SAE variants instead of 16k-width (for later gemma-3-12b-it only, ablation)")
@click.option("--no-optimizer-state", "no_optimizer_state", is_flag=True, default=False, help="Load model weights from checkpoint but start optimizer fresh (no resume)")
@click.option("--domain-sae-path", type=str, default=None,
              help="Path to a SparseCoder cache dir (from script_scoping_pipeline_stemqa_learnsae.py). "
                   "Replaces rank+prune with a pre-trained k-sparse SAE.")
@click.option("--hookpoint", "hookpoint_override", type=str, default=None,
              help="Override the default hookpoint (e.g. model.layers.38). Required when --domain-sae-path was trained at a non-default layer.")
@click.option("--skip-pre-training-eval", "skip_pre_training_eval", is_flag=True, default=False,
              help="Skip the pre-recover and pre-attack baseline evals (with SAE hooked in).")
def main(
    train_domain: str,
    attack_domain: str | None,
    stage: str,
    n_rank_samples: int,
    batch_size: int,
    accum: int,
    max_steps_recover: int,
    max_steps_attack: int,
    save_every: int,
    firing_rate_threshold: float,
    output_dir: str | None,
    checkpoint: str | None,
    hf_recover_repo: str | None,
    hf_attack_repo: str | None,
    device: str,
    use_gemma2: bool,
    use_gemma3: bool,
    use_olmo: bool,
    dev: bool,
    later_gemma3: bool,
    all_layers_recover: bool,
    _131k: bool,
    no_optimizer_state: bool,
    domain_sae_path: str | None,
    hookpoint_override: str | None,
    skip_pre_training_eval: bool,
):
    if sum([use_gemma2, use_gemma3, use_olmo]) > 1:
        raise click.UsageError("Specify at most one of --gemma2, --gemma3, --olmo.")
    if use_olmo:
        if domain_sae_path is None:
            raise click.UsageError("--domain-sae-path is required with --olmo (no Gemmascope SAE available).")
        cfg = OLMO_CONFIG
    elif _131k:
        cfg = GEMMA2_CONFIG_131K if use_gemma2 else GEMMA3_CONFIG_131K
        if later_gemma3:
            cfg = GEMMA3_LATER_CONFIG_131K
    else:
        cfg = GEMMA2_CONFIG if use_gemma2 else GEMMA3_CONFIG
        if later_gemma3:
            cfg = GEMMA3_LATER_CONFIG
    model_name = cfg["model_name"]
    sae_release = cfg["sae_release"]
    sae_id = cfg["sae_id"]
    hookpoint = hookpoint_override if hookpoint_override else cfg["hookpoint"]
    if hookpoint_override:
        import re as _re
        _lm = _re.search(r"layers\.(\d+)$", hookpoint)
        cache_tag = f"layer_{_lm.group(1)}" if _lm else cfg["cache_tag"]
    else:
        cache_tag = cfg["cache_tag"]

    device = torch.device(device)

    _valid_stages = {"all", "rank", "recover", "attack"}
    stages = {s.strip() for s in stage.split(",")}
    _invalid = stages - _valid_stages
    if _invalid:
        raise click.UsageError(f"Invalid stage(s): {_invalid}. Choose from: {_valid_stages}")
    if "all" in stages:
        stages = {"rank", "recover", "attack"}

    if "attack" in stages and attack_domain is None:
        raise click.UsageError("--attack-domain is required when stage includes 'attack'.")

    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")
    cache_dir = (
        base_dir / ".cache" / f"stemqa_{train_domain}"
        / "ignore_padding_True"
        / model_slug
        / cache_tag
        / f"n{n_rank_samples}"
    )
    # Shared eval dir: threshold-independent, so baseline_true.csv is computed once.
    # dev/prod use separate paths because they sample from different-sized question lists.
    shared_eval_dir = base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain / ("dev_llm_judge_csvs" if dev else "llm_judge_csvs")
    # ── Pre-load n_kept from cache (needed for output paths on attack stage) ─
    _dist_cache_path = cache_dir / "firing_rates.safetensors"
    if domain_sae_path is not None:
        # SparseCoder path: output path is based on SAE dim, not n_kept.
        _domain_sae_dir = Path(domain_sae_path)
        import json as _json
        with open(_domain_sae_dir / "sae" / "cfg.json") as _f:
            _domain_sae_cfg = _json.load(_f)
        n_kept = _domain_sae_cfg["num_latents"]
        output_base = Path(output_dir) if output_dir else (
            base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain
            / "domain_sae" / f"dh{n_kept}"
        )
    elif _dist_cache_path.exists():
        _pre = load_file(str(_dist_cache_path))
        n_kept = int((_pre["distribution"] >= firing_rate_threshold).sum().item())
        output_base = Path(output_dir) if output_dir else (
            base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain
            / f"h{firing_rate_threshold}" / f"k{n_kept}"
        )
    elif output_dir is not None:
        # Cache not yet computed, but an explicit output dir was given — use it so the
        # downstream recover_final.exists() check (attack stage) can do its job.
        output_base = Path(output_dir)
    elif stage == "attack":
        raise click.UsageError(
            f"Cannot determine output paths: no cached firing rates at {_dist_cache_path} "
            "and no --output-dir provided. Run --stage rank first, or pass --output-dir."
        )

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Load model ─────────────────────────────────────────────────────────
    attack_resume_from_checkpoint: bool | str = True
    if "attack" in stages:
        recover_final = output_base / "recover" / "final"
        if checkpoint is not None and checkpoint.isdigit():
            step = int(checkpoint)
            if hf_attack_repo is not None:
                print(f"Downloading checkpoint-{step} from HuggingFace {hf_attack_repo}...")
                local_dir = snapshot_download(
                    repo_id=hf_attack_repo,
                    allow_patterns=[f"checkpoint-{step}/*"],
                )
                checkpoint_local = str(Path(local_dir) / f"checkpoint-{step}")
                model_path = checkpoint_local
                attack_resume_from_checkpoint = checkpoint_local
                print(f"Resuming attack from step {step} (local: {checkpoint_local})")
            elif hf_recover_repo is not None:
                print(f"Downloading checkpoint-{step} from HuggingFace recover repo {hf_recover_repo}...")
                local_dir = snapshot_download(
                    repo_id=hf_recover_repo,
                    allow_patterns=[f"checkpoint-{step}/*"],
                )
                model_path = str(Path(local_dir) / f"checkpoint-{step}")
                print(f"Starting attack from recover checkpoint-{step} (local: {model_path})")
            else:
                raise click.UsageError(
                    "--hf-attack-repo or --hf-recover-repo is required when --checkpoint is a step number."
                )
        elif checkpoint:
            # Local checkpoint path (legacy / manual override).
            model_path = checkpoint
            attack_resume_from_checkpoint = checkpoint
            print(f"Resuming attack training from local checkpoint: {model_path}")
        elif hf_recover_repo:
            model_path = hf_recover_repo
            print(f"Loading post-recover model from HuggingFace: {model_path}")
        elif recover_final.exists():
            model_path = str(recover_final)
            print(f"Loading post-recover model from {model_path}")
        else:
            raise click.UsageError(
                f"No recover checkpoint found at {recover_final}. "
                "Run --stage recover first, pass --checkpoint, or pass --hf-recover-repo."
            )
    else:
        recover_resume_from_checkpoint: bool | str = True  # resolved after output_base is finalised
        if checkpoint is not None and checkpoint.isdigit():
            if hf_recover_repo is None:
                raise click.UsageError(
                    "--hf-recover-repo is required when --checkpoint is a step number for recover stage."
                )
            step = int(checkpoint)
            print(f"Downloading checkpoint-{step} from HuggingFace {hf_recover_repo}...")
            local_dir = snapshot_download(
                repo_id=hf_recover_repo,
                allow_patterns=[f"checkpoint-{step}/*"],
            )
            checkpoint_local = str(Path(local_dir) / f"checkpoint-{step}")
            model_path = checkpoint_local
            if no_optimizer_state:
                recover_resume_from_checkpoint = False
                print(f"Loading recover weights from checkpoint-{step} (no optimizer state)")
            else:
                recover_resume_from_checkpoint = checkpoint_local
                print(f"Resuming recover from step {step} (local: {checkpoint_local})")
        elif checkpoint:
            model_path = checkpoint
            if no_optimizer_state:
                recover_resume_from_checkpoint = False
                print(f"Loading recover weights from local checkpoint (no optimizer state): {model_path}")
            else:
                recover_resume_from_checkpoint = checkpoint
                print(f"Resuming recover training from local checkpoint: {model_path}")
        elif hf_recover_repo:
            model_path = hf_recover_repo
            recover_resume_from_checkpoint = False
            print(f"Loading recover model weights from HuggingFace (no optimizer state): {model_path}")
        else:
            model_path = model_name
            print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    # ── Load all domain datasets upfront (guarantees no train/eval leakage) ─
    # Coding is only loaded when it is the train or attack domain; it's expensive
    # to load and adds unnecessary eval overhead for non-coding experiments.
    _domains_to_load = [d for d in ALL_DOMAINS if d != "coding" or d in {train_domain, attack_domain}]
    print("Loading all domain datasets...")
    all_domain_splits: dict[str, tuple[Dataset, Dataset]] = {}
    for domain in _domains_to_load:
        t = time.time()
        tr, ev = load_domain_train_eval(domain, tokenizer)
        all_domain_splits[domain] = (tr, ev)
        print(f"  {domain}: {len(tr)} train, {len(ev)} eval in {time.time()-t:.1f}s")
    train_ds = all_domain_splits[train_domain][0]

    # ── Stage 1: RANK (skipped when --domain-sae-path is set) ────────────────
    ranking, distribution = None, None
    if domain_sae_path is None:
        if "rank" in stages:
            ranking, distribution = stage_rank(
                train_dataset=train_ds,
                n_samples=n_rank_samples,
                batch_size=batch_size,
                tokenizer=tokenizer,
                model=model,
                device=device,
                cache_dir=cache_dir,
                sae_release=sae_release,
                sae_id=sae_id,
                hookpoint=hookpoint,
            )
        else:
            cache_path = cache_dir / "firing_rates.safetensors"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"No cached firing rates at {cache_path}. Run with --stage rank first."
                )
            data = load_file(str(cache_path))
            ranking, distribution = data["ranking"], data["distribution"]

    # ── Finalise n_kept and output_base ───────────────────────────────────────
    if domain_sae_path is None:
        n_kept = int((distribution >= firing_rate_threshold).sum().item())
    if output_dir:
        output_base = Path(output_dir)
        recover_run_id = None
    else:
        recover_run_id = wandb.util.generate_id()
        if domain_sae_path is not None:
            output_base = (
                base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain
                / "domain_sae" / f"dh{n_kept}" / recover_run_id
            )
        else:
            output_base = (
                base_dir / "outputs_scoping" / model_slug / cache_tag / train_domain
                / f"h{firing_rate_threshold}" / f"k{n_kept}" / recover_run_id
            )

    # ── Build eval datasets ────────────────────────────────────────────────
    n_eval_cap = 500 if dev else None
    eval_datasets: dict[str, Dataset] = {}
    for domain, (_, ev) in all_domain_splits.items():
        eval_datasets[domain] = ev.select(range(min(n_eval_cap, len(ev)))) if n_eval_cap else ev
    print(f"Eval sizes ({'dev' if dev else 'prod'}): { {d: len(ev) for d, ev in eval_datasets.items()} }")

    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }
    domain_answers: dict[str, list[str]] = {
        name: ds["answer"] for name, ds in eval_datasets.items()
    }

    if domain_sae_path is not None:
        recover_run_name = f"recover/{model_slug}/{cache_tag}/{train_domain}/domain_sae/dh{n_kept}"
    else:
        recover_run_name = f"recover/{model_slug}/{cache_tag}/{train_domain}/h{firing_rate_threshold}/k{n_kept}"

    # ── Pre-init wandb for the recover run so its ID is embedded in output_base ─
    if "recover" in stages:
        init_kwargs: dict = dict(
            project=f"sae-scoping-stemqa-{train_domain}",
            name=recover_run_name,
            resume="allow",
            settings=wandb.Settings(init_timeout=180),
        )
        if recover_run_id is not None:
            init_kwargs["id"] = recover_run_id
        wandb.init(**init_kwargs)

    # ── True baseline eval (raw model, no SAE) ────────────────────────────
    _domain_gen_kwargs = {"coding": {"do_sample": False, "max_new_tokens": 1500}}

    if "recover" in stages or "attack" in stages:
        run_baseline_eval(
            model=model,
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            train_domain=train_domain,
            wandb_project=f"sae-scoping-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            csv_path=shared_eval_dir / "baseline_true.csv",
            metric_prefix="true_baseline",
            n_max_openai_requests=1_800,
            chart_suffix="pre_scoping",
            domain_answers=domain_answers,
            domain_generation_kwargs=_domain_gen_kwargs,
        )

    # ── Stage 2: PRUNE (skipped when --domain-sae-path is set) ───────────────
    raw_sae = None
    if domain_sae_path is not None:
        pruned_sae = _load_sae_from_cache(Path(domain_sae_path), device)
        print(f"Loaded SparseCoder from {domain_sae_path} (num_latents={pruned_sae.num_latents})")
    else:
        pruned_sae, raw_sae, n_kept = stage_prune(distribution, ranking, device, sae_release, sae_id, firing_rate_threshold)
    llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        domain_answers=domain_answers,
        llm_judge_every=500,
        n_max_openai_requests=1_800,
        model_name=model_name,
        run_name=recover_run_name,
        csv_dir=output_base / "llm_judge_csvs",
        train_domain=train_domain,
        reference_score_paths={
            "baseline": shared_eval_dir / "baseline_true.scores.json",
            "pre_recover": output_base / "llm_judge_csvs" / "baseline_pre_recover.scores.json",
        },
        domain_generation_kwargs=_domain_gen_kwargs,
    )

    # ── Stage 3: RECOVER ───────────────────────────────────────────────────
    # Resolve auto-detect sentinel: resume only if prior checkpoints exist in output_base/recover.
    if "recover" in stages and recover_resume_from_checkpoint is True:
        _recover_dir = output_base / "recover"
        recover_resume_from_checkpoint = _recover_dir.is_dir() and any(_recover_dir.glob("checkpoint-*"))

    if "recover" in stages:
        print("\n" + "=" * 80)
        print(f"STAGE 3: In-domain recovery training ({train_domain})")
        print("=" * 80)
        print(f"Train dataset: {len(train_ds)} samples")

        if not skip_pre_training_eval:
            run_baseline_eval(
                model=model,
                tokenizer=tokenizer,
                domain_questions=domain_questions,
                train_domain=train_domain,
                wandb_project=f"sae-scoping-stemqa-{train_domain}",
                wandb_run=recover_run_name,
                csv_path=output_base / "llm_judge_csvs" / "baseline_pre_recover.csv",
                metric_prefix="pre-recover-baseline",
                n_max_openai_requests=1_800,
                pruned_sae=pruned_sae,
                hookpoint=hookpoint,
                chart_suffix="post_scoping",
                domain_answers=domain_answers,
                domain_generation_kwargs=_domain_gen_kwargs,
            )

        recover_hf_cb = _HfCheckpointCallback()
        stage_train(
            train_dataset=train_ds,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(output_base / "recover"),
            wandb_project=f"sae-scoping-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            max_steps=max_steps_recover,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[llm_judge_callback, recover_hf_cb],
            resume_from_checkpoint=recover_resume_from_checkpoint,
            all_layers_after_hookpoint=all_layers_recover,
            max_seq_length=4096 if train_domain == "coding" else 1024,
        )
        save_path = str(output_base / "recover" / "final")
        print(f"Saving recover checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        recover_hf_cb.retry_failed_checkpoints()
        if recover_hf_cb.run_id is not None:
            recover_run_id = recover_hf_cb.run_id
            recover_dir = output_base / "recover"
            print(f"Uploading recover final model to HuggingFace Hub as {recover_run_id!r}...")
            try:
                model.push_to_hub(recover_run_id)
                tokenizer.push_to_hub(recover_run_id)
                print(f"Deleting local recover dir at {recover_dir}...")
                shutil.rmtree(recover_dir)
            except Exception as e:
                sys.exit(f"ERROR: HuggingFace final recover model upload failed ({e}).")

    # ── Stage 4: ATTACK ───────────────────────────────────────────────────
    if "attack" in stages:
        # End the recover wandb run (if active) before starting the attack run.
        if wandb.run is not None:
            wandb.finish()
        print("\n" + "=" * 80)
        print(f"STAGE 4: Adversarial elicitation ({attack_domain})")
        print("=" * 80)

        attack_run_id = wandb.util.generate_id()
        attack_output_base = output_base / "attack" / attack_domain / attack_run_id
        if domain_sae_path is not None:
            attack_run_name = f"attack/{model_slug}/{cache_tag}/{train_domain}/domain_sae/dh{n_kept}/{attack_domain}"
        else:
            attack_run_name = f"attack/{model_slug}/{cache_tag}/{train_domain}/h{firing_rate_threshold}/k{n_kept}/{attack_domain}"
        wandb.init(
            project=f"sae-scoping-stemqa-{train_domain}",
            name=attack_run_name,
            id=attack_run_id,
            resume="allow",
            settings=wandb.Settings(init_timeout=180),
        )
        _attack_domains = {train_domain, attack_domain}
        attack_domain_questions = {k: v for k, v in domain_questions.items() if k in _attack_domains}
        attack_domain_answers = {k: v for k, v in domain_answers.items() if k in _attack_domains} if domain_answers else None
        attack_eval_datasets = {k: v for k, v in eval_datasets.items() if k in _attack_domains}

        attack_llm_judge_callback = LLMJudgeScopingTrainerCallback(
            tokenizer=tokenizer,
            domain_questions=attack_domain_questions,
            domain_answers=attack_domain_answers,
            llm_judge_every=500,
            n_max_openai_requests=1_800,
            model_name=model_name,
            run_name=attack_run_name,
            csv_dir=attack_output_base / "llm_judge_csvs",
            train_domain=train_domain,
            attack_domain=attack_domain,
            reference_score_paths={
                "baseline": shared_eval_dir / "baseline_true.scores.json",
                "pre_attack": attack_output_base / "llm_judge_csvs" / "baseline_pre_attack.scores.json",
            },
            domain_generation_kwargs=_domain_gen_kwargs,
        )

        adversarial_dataset = all_domain_splits[attack_domain][0]
        print(f"Attack dataset: {len(adversarial_dataset)} train samples ({attack_domain})")

        if not skip_pre_training_eval:
            run_baseline_eval(
                model=model,
                tokenizer=tokenizer,
                domain_questions=attack_domain_questions,
                train_domain=train_domain,
                attack_domain=attack_domain,
                wandb_project=f"sae-scoping-stemqa-{train_domain}",
                wandb_run=attack_run_name,
                csv_path=attack_output_base / "llm_judge_csvs" / "baseline_pre_attack.csv",
                metric_prefix="pre-attack-baseline",
                n_max_openai_requests=1_800,
                chart_suffix="pre_attack",
                domain_answers=attack_domain_answers,
                pruned_sae=pruned_sae,
                hookpoint=hookpoint,
                domain_generation_kwargs=_domain_gen_kwargs,
            )

        attack_hf_cb = _HfCheckpointCallback()
        stage_train(
            train_dataset=adversarial_dataset,
            eval_datasets=attack_eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(attack_output_base),
            wandb_project=f"sae-scoping-stemqa-{train_domain}",
            wandb_run=attack_run_name,
            max_steps=max_steps_attack,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[attack_llm_judge_callback, attack_hf_cb],
            all_layers_after_hookpoint=True,
            resume_from_checkpoint=attack_resume_from_checkpoint,
            max_seq_length=4096 if attack_domain == "coding" else 1024,
        )
        save_path = str(attack_output_base / "final")
        print(f"Saving attack checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        attack_hf_cb.retry_failed_checkpoints()
        if attack_hf_cb.run_id is not None:
            run_id = attack_hf_cb.run_id
            print(f"Uploading attack final model to HuggingFace Hub as {run_id!r}...")
            try:
                model.push_to_hub(run_id)
                tokenizer.push_to_hub(run_id)
                print(f"Deleting local attack dir at {attack_output_base}...")
                shutil.rmtree(attack_output_base)
            except Exception as e:
                sys.exit(f"ERROR: HuggingFace final attack model upload failed ({e}).")
            # Upload wandb run directory (do not delete it locally).
            api = HfApi()
            wandb_run_dirs = list((base_dir / "wandb").glob(f"run-*-{run_id}"))
            if wandb_run_dirs:
                wandb_run_dir = wandb_run_dirs[0]
                print(f"Uploading wandb files from {wandb_run_dir.name} to HuggingFace Hub {run_id!r}...")
                try:
                    api.upload_folder(
                        folder_path=str(wandb_run_dir),
                        repo_id=run_id,
                        path_in_repo=f"wandb/{wandb_run_dir.name}",
                        repo_type="model",
                    )
                except Exception as e:
                    print(f"Warning: failed to upload wandb dir ({e}); skipping.")

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model, pruned_sae
    if raw_sae is not None:
        del raw_sae
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
