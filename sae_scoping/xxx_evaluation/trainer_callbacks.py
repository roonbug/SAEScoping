from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
import io
import orjson
import json
import torch
import wandb
import pandas as pd
from transformers import TrainerCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from beartype import beartype
from sae_scoping.xxx_evaluation.spylab_1click_judgement import (
    OneClickLLMJudgeEvaluationETHZ1Biology,
)
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval


# XXX clean this up a lot plz
class LLMJudgeSpylabBio1ClickTrainerCallback(TrainerCallback):
    """
    This callback adds custom metrics-update logic (what you see on CLI or WanDB or
    whatever your logging strategy is) to evaluate the model using the LLM judge in such
    a way that we can get more realistic sense of whether the LLM is good.

    Mostly a refactor by claude.
    """

    @beartype
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        llm_judge_every: int = 100,
        save_full_info: bool = False,
        save_full_info_mode: str = "wandb",
        model_name: str = "unknown",
        run_name: str = "unknown",
        full_info_folder: Optional[Path] = None,
    ):
        self.llm_judge_every = llm_judge_every
        self.save_full_info = save_full_info
        self.save_full_info_mode = save_full_info_mode
        self.full_info_folder = full_info_folder
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.run_name = run_name
        self.evaluator = OneClickLLMJudgeEvaluationETHZ1Biology(
            n_max_openai_requests=200_000  # in my experience OK
        )

        # Validate settings"
        if self.save_full_info and self.save_full_info_mode == "folder":
            assert self.full_info_folder is not None, (
                "full_info_folder must be provided when mode is 'folder'"
            )
            assert not (
                self.full_info_folder.exists()
                or len(list(self.full_info_folder.iterdir())) == 0
            ), f"File {self.full_info_folder} already exists"
            self.full_info_folder.parent.mkdir(parents=True, exist_ok=True)
        elif self.save_full_info and self.save_full_info_mode not in [
            "folder",
            "wandb",
            "wandb_conservative",
        ]:
            raise ValueError(f"Invalid save_full_info_mode: {self.save_full_info_mode}")

    @classmethod
    @beartype
    def _evaluate(
        cls,
        model,
        tokenizer,
        evaluator,
        global_step: Any,
        save_full_info: bool,
        save_full_info_mode: str,
        full_info_folder: Optional[Path],
        model_name: str,
        run_name: str,
        # Replace so that we can have special panels for eval before training etc...
        replace_llm_judge_with: str = "llm_judge",
    ) -> dict[Any, Any]:
        """
        Static/class function to do most of the work so we can also call
        externally (kinda janky but eh whatever).
        """
        metrics = {}
        print("@" * 100)
        print("@" * 40 + " STARTING CUSTOM EVALUATION (LLM JUDGE) " + "@" * 40)
        print(f"\n🔍 Running LLM judge evaluation at step {global_step}...")

        assert replace_llm_judge_with.startswith("llm_judge")
        with torch.no_grad():
            (
                custom_results_utility_and_safety,
                custom_results_utility_and_safety_df_as_json,
            ) = evaluator.evaluate(model, tokenizer, n_max_openai_requests=1_000)
            custom_results_utility_and_safety = {
                k.replace("llm_judge", replace_llm_judge_with): v
                for k, v in custom_results_utility_and_safety.items()
            }
            assert all(
                k.startswith("llm_judge")
                for k in custom_results_utility_and_safety.keys()
            )

            metrics.update(custom_results_utility_and_safety)
            if wandb.run is not None:
                wandb.log(
                    {
                        **{k: v for k, v in custom_results_utility_and_safety.items()},
                        "trainer/global_step": global_step,
                    }
                )

            # Save full info if requested
            if save_full_info:
                if save_full_info_mode == "folder":
                    file = (
                        full_info_folder
                        / f"{replace_llm_judge_with}_{global_step}.json"
                    )
                    assert not file.exists()  # should never happen....
                    file.write_text(custom_results_utility_and_safety_df_as_json)
                elif save_full_info_mode == "wandb":
                    # Add to metrics (will be logged to wandb)
                    vizkey = f"{replace_llm_judge_with}/full_info_json"
                    assert vizkey.startswith(replace_llm_judge_with)
                    if vizkey in metrics:
                        print(f"⚠️  Warning: '{vizkey}' already in metrics")
                    metrics[vizkey] = custom_results_utility_and_safety_df_as_json
                    if wandb.run is not None:
                        wandb.log(
                            {
                                **{
                                    vizkey: custom_results_utility_and_safety_df_as_json
                                },
                                "trainer/global_step": global_step,
                            }
                        )
                elif save_full_info_mode == "wandb_conservative":
                    # In this case we parse the json, extract only the generations, and
                    # then report only that basically
                    # NOTE we have a key per model/run and seperate from numerical
                    # metrics purely for easier visualization
                    # NOTE: run name is too long so we don't use :( table should be
                    # enough maybe?
                    vizkey = f"{replace_llm_judge_with}_generations/{model_name}/{global_step}"
                    assert vizkey.startswith("llm_judge")  # this is still important
                    if vizkey in metrics:
                        print(f"⚠️  Warning: {vizkey} already in metrics")
                    generations_pd: list[dict[str, Any]] = orjson.loads(
                        custom_results_utility_and_safety_df_as_json.encode()
                    )
                    assert isinstance(generations_pd, list)
                    assert all(isinstance(x, dict) for x in generations_pd)
                    expected_keys = set(
                        [
                            # NOTE: this should match schema from pydantic in
                            # the code `spylab_1click_judgement.py`
                            "prompt",
                            "response",
                            "seed",
                            "judge_name",
                            "judge_template",
                            "judgement_score",
                            "judgement_explanation",
                        ]
                    )
                    assert all(
                        set(x.keys()) == expected_keys for x in generations_pd
                    ), (
                        "key-sets were "
                        + f"{set(frozenset(z.keys()) for z in generations_pd)}"
                    )

                    print(f"DEBUG: generations_pd has {len(generations_pd)} items")
                    print(
                        f"DEBUG: wandb.run is {'active' if wandb.run is not None else 'None'}"
                    )
                    # print(f"DEBUG: First few generations: {generations_pd[:2] if generations_pd else 'EMPTY'}")
                    # wandb table is a recommendation from Claude
                    generation_table = wandb.Table(
                        columns=[
                            "user_request",
                            "assistant_response",
                            "model_name",
                            "run_name",
                            "global_step",
                        ],
                        data=[
                            [
                                g["prompt"],
                                g["response"],
                                model_name,
                                run_name,
                                global_step,
                            ]
                            for g in generations_pd
                        ],
                    )
                    # NOTE: we report the generations that are serializeable BUT we
                    # log the TABLE to wandb
                    metrics[vizkey] = generations_pd
                    if wandb.run is not None:
                        wandb.log(
                            {
                                **{vizkey: generation_table},
                                "trainer/global_step": global_step,
                            }
                        )
                else:
                    print("WARNING: someone messed with mode wtf")
                print("LLM Judge metrics!")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in metrics.items()
                            if k.startswith("llm_judge")
                            and isinstance(v, (float, bool, int))
                        }
                    )
                )
        print("@" * 100)
        assert metrics is not None  # shoulda short-circuited
        # TODO(Adriano) for unknown reasons this was not being logged to wandb (i.e.
        # callback was too late for wandb; I need to read docs; hotfix with
        # wandb.log above...)
        print("Final metrics being logged:\n - " + "\n - ".join(metrics.keys()))
        print("...")
        print("@" * 40 + " FINISHED CUSTOM EVALUATION (LLM JUDGE) " + "@" * 40)
        print("@" * 100)
        return metrics

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Called after evaluation - add custom LLM judge metrics"""
        # Check if we should run the LLM judge (on first eval or at specified intervals)
        # Note: state.global_step tracks training steps
        if state.global_step == 0 or state.global_step % self.llm_judge_every == 0:
            if metrics is None:  # NOTE: metrics needs to be UPDATED acc. to claude
                print("WARNING: METRICS IS NONE; CALLBACK WILL DO NOTHING")
                return
            # Make sure no key collision
            assert not any(
                isinstance(k, str) and k.startswith("llm_judge") for k in metrics.keys()
            ), str(set(metrics.keys()))
            # Collect the new metrics
            new_metrics2log = self._evaluate(
                model,
                self.tokenizer,
                self.evaluator,
                state.global_step,
                self.save_full_info,
                self.save_full_info_mode,
                self.full_info_folder,
                self.model_name,
                self.run_name,
            )
            # Make sure again no collision
            assert all(
                isinstance(k, str) and k.startswith("llm_judge")
                for k in new_metrics2log.keys()
            ), str(set(new_metrics2log.keys()))
            # Update for logging with huggingface
            metrics.update(new_metrics2log)


class LLMJudgeScopingTrainerCallback(TrainerCallback):
    """
    TrainerCallback that runs OneClickLLMJudgeScopingEval during training and
    logs results to W&B. Mirrors LLMJudgeSpylabBio1ClickTrainerCallback but
    uses the domain-based scoping evaluator (no trojans).
    """

    @beartype
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        domain_questions: dict[str, list[str]],
        llm_judge_every: int = 100,
        n_max_openai_requests: int = 1_000,
        model_name: str = "unknown",
        run_name: str = "unknown",
        csv_dir: Optional[Path] = None,
        train_domain: Optional[str] = None,
        attack_domain: Optional[str] = None,
        domain_answers: Optional[dict[str, list[str]]] = None,
        reference_score_paths: Optional[dict[str, Path]] = None,
        domain_generation_kwargs: Optional[dict[str, dict]] = None,
    ):
        self.tokenizer = tokenizer
        self.domain_questions = domain_questions
        self.domain_answers = domain_answers
        self.llm_judge_every = llm_judge_every
        self.n_max_openai_requests = n_max_openai_requests
        self.model_name = model_name
        self.run_name = run_name
        self.csv_dir = csv_dir
        self._step_scores: list[dict[str, float]] = []
        self._step_dfs: list[pd.DataFrame] = []
        self._current_step: int = -1
        self._call_index: int = 0
        self.n_eval_datasets: int = len(domain_questions)
        self.n_eval_runs: int = 1
        self.evaluator = OneClickLLMJudgeScopingEval(
            n_max_openai_requests=200_000,
            train_domain=train_domain,
            attack_domain=attack_domain,
            domain_generation_kwargs=domain_generation_kwargs or {},
        )
        # History for grouped line-series charts (one chart per judge type).
        self._eval_steps: list[int] = []
        self._score_history: dict[str, list[float]] = {}
        # Paths to pre-computed baseline score JSON files (label → path).
        self._reference_score_paths: dict[str, Path] = reference_score_paths or {}
        self._reference_scores_cache: dict[str, dict[str, float]] = {}

    def _load_reference_scores(self) -> dict[str, dict[str, float]]:
        """Read each reference scores.json file on first use; cache results."""
        for label, path in self._reference_score_paths.items():
            if label not in self._reference_scores_cache and path.exists():
                try:
                    self._reference_scores_cache[label] = json.loads(path.read_text())
                except Exception as e:
                    print(f"Warning: could not load reference scores from {path}: {e}")
        return self._reference_scores_cache

    def _log_grouped_charts(self, step: int) -> None:
        """Log one wandb line_series chart per judge type, with all domains as lines."""
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for k in sorted(self._score_history.keys()):
            judge_type = k.split("/")[-1]
            groups[judge_type].append(k)
        for judge_type, keys in sorted(groups.items()):
            xs = [self._eval_steps[:] for _ in keys]
            ys = [self._score_history[k][:] for k in keys]
            labels = ["/".join(k.split("/")[1:3]) for k in keys]  # e.g. "biology/in_scope"
            chart = wandb.plot.line_series(
                xs=xs, ys=ys, keys=labels,
                title=f"LLM Judge: {judge_type}",
                xname="Training Step",
            )
            wandb.log({f"charts/llm_judge_{judge_type}": chart, "trainer/global_step": step})

        ref_scores = self._load_reference_scores()
        for label, ref in ref_scores.items():
            for judge_type, keys in sorted(groups.items()):
                def _ref_key(k):
                    if k in ref:
                        return k
                    fallback = k.replace("attack_scope", "out_of_scope")
                    return fallback if fallback in ref else None
                filtered_keys = [k for k in keys if _ref_key(k) is not None]
                ys = [
                    [v - ref[_ref_key(k)] for v in self._score_history[k]]
                    for k in filtered_keys
                ]
                labels = ["/".join(k.split("/")[1:3]) for k in filtered_keys]
                xs = [self._eval_steps[:] for _ in filtered_keys]
                if not ys:
                    continue
                chart = wandb.plot.line_series(
                    xs=xs, ys=ys, keys=labels,
                    title=f"LLM Judge diff vs {label}: {judge_type}",
                    xname="Training Step",
                )
                wandb.log({f"charts/llm_judge_diff_{label}_{judge_type}": chart, "trainer/global_step": step})

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if state.global_step % self.llm_judge_every != 0:
            return
        if metrics is None:
            print("WARNING: METRICS IS NONE; LLMJudgeScopingTrainerCallback will do nothing")
            return

        # Reset accumulators when we enter a new step
        if state.global_step != self._current_step:
            self._current_step = state.global_step
            self._step_scores = []
            self._step_dfs = []
            self._call_index = 0

        self._call_index += 1
        call_idx = self._call_index

        if call_idx > self.n_eval_runs:
            return

        print("@" * 80)
        print(f"Running scoping LLM judge evaluation at step {state.global_step} (run {call_idx}/{self.n_eval_runs})...")
        if self.csv_dir is not None:
            self.evaluator.judge_inputs_save_dir = self.csv_dir
        with torch.no_grad():
            scores, df_as_json = self.evaluator.evaluate(
                model,
                self.tokenizer,
                self.domain_questions,
                n_max_openai_requests=self.n_max_openai_requests,
                domain_answers=self.domain_answers,
            )
        self._step_scores.append(scores)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        self._step_dfs.append(df)

        print(json.dumps({k: v for k, v in scores.items()}))

        # Save per-run CSV
        if self.csv_dir is not None:
            self.csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self.csv_dir / f"llm_judge_step_{state.global_step}_run{call_idx}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved judgements CSV to {csv_path}")

        # Log individual run to W&B (only meaningful when there are multiple runs)
        if wandb.run is not None and self.n_eval_runs > 1:
            wandb.log({**{f"{k}_run{call_idx}": v for k, v in scores.items()}, "trainer/global_step": state.global_step})

        # After all runs: finalize scores, log charts
        if call_idx == self.n_eval_runs:
            if self.n_eval_runs > 1:
                avg_scores = {
                    k: float(sum(s[k] for s in self._step_scores) / len(self._step_scores))
                    for k in self._step_scores[0]
                }
                print("Averaged scores:")
                print(json.dumps(avg_scores))

                if self.csv_dir is not None:
                    all_df = pd.concat(self._step_dfs, ignore_index=True)
                    avg_csv_path = self.csv_dir / f"llm_judge_step_{state.global_step}_avg.csv"
                    all_df.to_csv(avg_csv_path, index=False)
                    print(f"Saved averaged judgements CSV to {avg_csv_path}")

                if wandb.run is not None:
                    wandb.log({**{f"{k}_avg": v for k, v in avg_scores.items()}, "trainer/global_step": state.global_step})
                metrics.update({f"{k}_avg": v for k, v in avg_scores.items()})
            else:
                avg_scores = self._step_scores[0]
                if wandb.run is not None:
                    wandb.log({**avg_scores, "trainer/global_step": state.global_step})
                metrics.update(avg_scores)

            if wandb.run is not None:
                if self._step_dfs:
                    all_df = pd.concat(self._step_dfs, ignore_index=True)
                    wandb.log({"llm_judge/judgements": wandb.Table(dataframe=all_df), "trainer/global_step": state.global_step})
                self._eval_steps.append(state.global_step)
                for k, v in avg_scores.items():
                    self._score_history.setdefault(k, []).append(v)
                self._log_grouped_charts(state.global_step)

                # Diff scalars: one set per reference baseline.
                for label, ref in self._load_reference_scores().items():
                    diffs = {
                        k.replace("llm_judge/", f"llm_judge_diff_{label}/"): v - ref[k]
                        for k, v in avg_scores.items()
                        if k in ref
                    }
                    if diffs:
                        wandb.log({**diffs, "trainer/global_step": state.global_step})

        print("@" * 80)
        torch.cuda.empty_cache()
