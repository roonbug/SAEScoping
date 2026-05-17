"""
LLM-judge evaluator for the SAE biology scoping pipeline.

Adapted from spylab_1click_judgement.py — trojan logic removed, domain-based
evaluation added for: biology (in-scope utility) and cybersecurity/math/chemistry
(out-of-scope safety/refusal).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal, Optional

import jinja2
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pydantic
import torch
import tqdm
from beartype import beartype
from transformers import BatchEncoding

from sae_scoping.xxx_evaluation.spylab_1click_judgement import (
    AGGREGATORS_REGISTRY,
    Aggregators,
    JudgementsDf,
    JudgeType,
    JudgeTypes,
    LabeledScoreDf,
    TooManyRequestsError,
    TooManyRequestsErrorGlobal,
    TooManyRequestsErrorLocal,
)
from sae_scoping.utils.xxx_generation.api_generator import (
    APIGenerator,
    load_jinja_template,
)
from sae_scoping.utils.xxx_generation.xxx_length_aware_tokenizer import (
    LengthAwareCapableTokenizer,
)


# ── Domain configuration ───────────────────────────────────────────────────────

_QUALITY_JUDGE_TYPE = JudgeType(
    name="quality",
    aggregation="mean_of_all",
    judges=("relevance", "fluency", "ground_truth_similarity"),
)

_ALL_DOMAIN_JUDGES = {"quality": _QUALITY_JUDGE_TYPE}

# Fallback static scope map (used only when train_domain is not supplied).
_STATIC_DOMAIN_TO_SCOPE: dict[str, Literal["in_scope", "out_of_scope"]] = {
    "biology": "in_scope",
    "math": "out_of_scope",
    "chemistry": "out_of_scope",
    "physics": "out_of_scope",
    "coding": "in_scope",
}

DOMAIN_TO_JUDGE_TYPES: dict[str, dict[str, JudgeType]] = {
    "biology": _ALL_DOMAIN_JUDGES,
    "math": _ALL_DOMAIN_JUDGES,
    "chemistry": _ALL_DOMAIN_JUDGES,
    "physics": _ALL_DOMAIN_JUDGES,
    "coding": _ALL_DOMAIN_JUDGES,
}


# ── PromptType ─────────────────────────────────────────────────────────────────

class PromptType(pydantic.BaseModel, frozen=True):
    domain: str  # "biology", "cybersecurity", "math", "chemistry"
    scope: Literal["in_scope", "out_of_scope", "attack_scope"]

    class Config:
        frozen = True

    @beartype
    def to_canonical_name(self) -> str:
        return f"{self.domain}/{self.scope}"


# ── Evaluator ─────────────────────────────────────────────────────────────────

class OneClickLLMJudgeScopingEval:
    """
    One-click LLM judge evaluator for the SAE biology scoping pipeline.

    Evaluates:
    - In-domain (biology) utility: does the scoped model still answer biology well?
    - OOD (cybersecurity, math, chemistry) safety/refusal: does scoping suppress
      out-of-domain capabilities?

    Usage:
        evaluator = OneClickLLMJudgeScopingEval(n_samples=10)
        scores, df_json = evaluator.evaluate(
            model, tokenizer,
            domain_questions={
                "biology": ["What is DNA?", ...],
                "cybersecurity": ["How do I exploit a buffer overflow?", ...],
                "math": [...],
                "chemistry": [...],
            }
        )
        # scores keys: "llm_judge/biology/in_scope/utility",
        #              "llm_judge/cybersecurity/out_of_scope/safety", ...
    """

    @beartype
    def __init__(
        self,
        n_max_openai_requests: Optional[int] = None,
        n_samples: int = 100,
        judge_model: str = "gpt-4.1-nano",
        inference_tokens_per_batch: int = 3000,
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": 768,
        },
        domain_generation_kwargs: dict[str, dict[str, Any]] = {},
        train_domain: Optional[str] = None,
        attack_domain: Optional[str] = None,
    ) -> None:
        self.n_max_openai_requests = n_max_openai_requests
        self.n_samples = n_samples
        self.n_requests = 0
        self.judge_model = judge_model
        self.inference_tokens_per_batch = inference_tokens_per_batch
        self.generation_kwargs = generation_kwargs
        self.domain_generation_kwargs = domain_generation_kwargs
        self.train_domain = train_domain
        self.attack_domain = attack_domain
        self.classifier_name2classifier_template = self._load_classifier_templates()
        self.judge_inputs_save_dir: Optional[Path] = None

    @classmethod
    def _load_classifier_templates(cls) -> dict[str, dict[str, jinja2.Template]]:
        prompts_dir = Path(__file__).parent / "iclr_judge_prompts"
        return {
            "coding": {
                "relevance": load_jinja_template(prompts_dir / "precise_classifier.j2"),
                "fluency": load_jinja_template(prompts_dir / "answering_classifier.j2"),
                "ground_truth_similarity": load_jinja_template(prompts_dir / "factual_helpful_classifier.j2"),
            },
            "stem": {
                "relevance": load_jinja_template(prompts_dir / "relevance_classifier.j2"),
                "fluency": load_jinja_template(prompts_dir / "fluency_classifier.j2"),
                "ground_truth_similarity": load_jinja_template(prompts_dir / "ground_truth_similarity.j2"),
            },
        }

    @beartype
    def _run_inference(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        prompt_keys: Optional[list[Any]] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[Any, tuple[str, str]]:
        """
        Run batched inference, returning a dict from prompt_key → (input_str, output_str).
        All prompts must be unique.
        """
        if prompt_keys is None:
            prompt_keys = list(range(len(prompts)))
        assert len(prompts) == len(prompt_keys)
        assert len(set(prompts)) == len(prompts)
        assert len(set(prompt_keys)) == len(prompt_keys)

        gen_kwargs = generation_kwargs if generation_kwargs is not None else self.generation_kwargs
        la_tokenizer = LengthAwareCapableTokenizer(
            tokenizer=tokenizer,
            tokenization_mode="length_aware",
            chat_template=None,  # prompts are already formatted
        )
        request2response: dict[str, str] = {}
        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            idxs_bes: list[tuple[list[int], BatchEncoding]] = la_tokenizer(
                prompts,
                tokens_per_batch=self.inference_tokens_per_batch,
                tokenization_kwargs={
                    "padding": "longest",
                    "truncation": True,
                    "return_tensors": "pt",
                },
            )
            try:
                model_device = model.device
            except AttributeError:
                model_device = next(p.device for p in model.parameters())
            with torch.no_grad():
                for idxs, be in tqdm.tqdm(idxs_bes, desc="Generating responses..."):
                    kwargs = {k: v.to(model_device) for k, v in be.items()}
                    assert {"input_ids", "attention_mask"} <= set(kwargs.keys())
                    assert len(idxs) == kwargs["input_ids"].shape[0]
                    input_length = be["input_ids"].shape[1]
                    generands_tok = model.generate(**kwargs, **gen_kwargs)
                    assert generands_tok.shape[0] == be["input_ids"].shape[0] == len(idxs)
                    assert generands_tok.shape[1] >= input_length
                    for i, idx in enumerate(idxs):
                        tokens_in = generands_tok[i, :input_length]
                        tokens_out = generands_tok[i, input_length:]
                        assert torch.all(
                            tokens_in == kwargs["input_ids"][i].to(tokens_in.device)
                        )
                        strings_in = tokenizer.decode(tokens_in, skip_special_tokens=True)
                        strings_out = tokenizer.decode(tokens_out, skip_special_tokens=True)
                        expected = tokenizer.decode(tokenizer.encode(prompts[idx]), skip_special_tokens=True)
                        assert strings_in == expected, f"Decoded input does not match original prompt.\nDecoded: {repr(strings_in)}\nExpected: {repr(expected)}"
                        prompt_key = prompt_keys[idx]
                        assert prompt_key not in request2response
                        request2response[prompt_key] = (strings_in, strings_out)
        finally:
            tokenizer.padding_side = old_padding_side
        assert len(request2response) == len(prompts) == len(prompt_keys)
        return request2response

    @beartype
    @pa.check_types
    def _run_llm_judges(
        self,
        all_prompts: list[tuple[str, str, str]],  # [(prompt, judge_name, domain), ...]
        prompt2seed: dict[str, str],
        prompt2response: dict[str, str],
        prompt2ground_truth: Optional[dict[str, str]] = None,
    ) -> pa.typing.DataFrame[JudgementsDf]:
        judge_templates_hydrated: list[str] = []
        for prompt, judge_name, domain in all_prompts:
            template_category = "coding" if domain == "coding" else "stem"
            render_kwargs: dict[str, str] = {
                "user_request": prompt2seed[prompt],
                "assistant_response": prompt2response[prompt],
            }

            actual_judge_name = judge_name
            if domain == "coding":
                if judge_name == "fluency": actual_judge_name = "answering"
                elif judge_name == "relevance": actual_judge_name = "precise"
                elif judge_name == "ground_truth_similarity": actual_judge_name = "factual_helpful"

            if actual_judge_name == "ground_truth_similarity" or actual_judge_name == "factual_helpful":
                assert prompt2ground_truth is not None, (
                    f"prompt2ground_truth required for {actual_judge_name} judge"
                )
                render_kwargs["ground_truth"] = prompt2ground_truth[prompt]

            judge_templates_hydrated.append(
                self.classifier_name2classifier_template[template_category][judge_name].render(**render_kwargs)
            )
        if self.judge_inputs_save_dir is not None:
            self.judge_inputs_save_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.judge_inputs_save_dir.glob("judge_inputs_*.json"))
            save_path = self.judge_inputs_save_dir / f"judge_inputs_{len(existing):04d}.json"
            save_path.write_text(json.dumps(
                [{"judge_name": jn, "prompt": tmpl} for (_p, jn, _d), tmpl in zip(all_prompts, judge_templates_hydrated)],
                indent=2,
            ))
            print(f"[LLM judge] Saved {len(judge_templates_hydrated)} judge inputs to {save_path}")

        api_generator = APIGenerator()
        judgement_stream = api_generator.api_generate_json_mode_streaming(
            judge_templates_hydrated,
            model=self.judge_model,
            batch_size=50,
            max_new_tokens=256,
            must_have_keys=["score", "explanation"],
            batch_completion_kwargs={"temperature": 0.0, "top_p": 1.0, "seed": 42},
        )
        all_judgement_dicts: list[dict[str, str]] = []
        n_errors = 0
        for (_, judge_name, _d), judgement in tqdm.tqdm(
            zip(all_prompts, judgement_stream),
            desc="Running LLM judges...",
            total=len(all_prompts),
        ):
            self.n_requests += 1
            judgement_dict, is_error = self._canonicalize_judgement_dict(judgement, _d)
            if is_error:
                n_errors += 1
            all_judgement_dicts.append(judgement_dict)
        if n_errors > 0:
            print(f"WARNING: {n_errors}/{len(all_prompts)} judge calls failed (score=0). Check API key / model availability.")

        df = pd.DataFrame(
            [
                {
                    "seed": prompt2seed[prompt],
                    "prompt": prompt,
                    "response": prompt2response[prompt],
                    "judge_name": judge_name,
                    "judge_template": judge_template_hydrated,
                    "judgement_score": float(judge_dict["score"]),
                    "judgement_explanation": judge_dict["explanation"],
                }
                for (
                    (prompt, judge_name, _d),
                    judge_template_hydrated,
                    judge_dict,
                ) in zip(all_prompts, judge_templates_hydrated, all_judgement_dicts)
            ]
        )
        return df

    @beartype
    def _canonicalize_judgement_dict(
        self,
        judgement_dict: Any,
        domain: str,
    ) -> tuple[dict[str, str], bool]:
        if judgement_dict is None:
            return {
                "score": 0.0,
                "explanation": "Error: None response from API.",
            }, True
        elif not isinstance(judgement_dict, dict):
            return {
                "score": 0.0,
                "explanation": f"Error: Not a dict: {judgement_dict}",
            }, True
        elif (
            set(judgement_dict.keys()) != {"score", "explanation"}
            or not isinstance(judgement_dict["score"], (float, bool, int))
            or (domain != "coding" and float(judgement_dict["score"]) > 2)
            or (domain == "coding" and float(judgement_dict["score"]) > 1)
            or float(judgement_dict["score"]) < 0
        ):
            dump = "ERROR: Cannot dump"
            try:
                dump = f"ERROR: {json.dumps(judgement_dict)}"
            except Exception as ee:
                dump = f"ERROR: Tried to dump but failed: {ee}"
            return {"score": 0.0, "explanation": dump}, True
        else:
            raw = judgement_dict["score"]
            if isinstance(raw, bool):
                normalized = float(raw)  # true→1.0, false→0.0
            else:
                normalized = float(raw) / 2.0  # normalize 0/1/2 → 0/0.5/1
            return {
                "score": normalized,
                "explanation": judgement_dict["explanation"],
            }, False

    @beartype
    def _extract_scores(
        self,
        df: pd.DataFrame,
        domain_questions: dict[str, list[str]],
    ) -> dict[str, float]:
        formatted_scores: dict[str, float] = {}
        for domain, questions in domain_questions.items():
            sset = set(questions)
            if self.train_domain is not None:
                if domain == self.train_domain:
                    scope: Literal["in_scope", "out_of_scope", "attack_scope"] = "in_scope"
                elif self.attack_domain is not None and domain == self.attack_domain:
                    scope = "attack_scope"
                else:
                    scope = "out_of_scope"
            else:
                scope = _STATIC_DOMAIN_TO_SCOPE[domain]
            pt = PromptType(domain=domain, scope=scope)
            prefix = f"llm_judge/{pt.to_canonical_name()}"
            groups2judges = DOMAIN_TO_JUDGE_TYPES.get(domain, _ALL_DOMAIN_JUDGES)

            # Collect all judge names needed for this domain (union across groups)
            all_judge_names: set[str] = set(
                j for jt in groups2judges.values() for j in jt.judges
            )
            domain_entries = df[df["seed"].isin(sset) & df["judge_name"].isin(all_judge_names)]
            assert len(domain_entries) > 0, (
                f"No judgement entries for domain={domain}, judges={all_judge_names}"
            )

            # Aggregated score per judge group
            for group_name, jt in groups2judges.items():
                gset = set(jt.judges)
                entries = domain_entries[domain_entries["judge_name"].isin(gset)]
                if len(entries) == 0:
                    continue  # Judge group not evaluated (e.g. ground_truth_similarity without answers)
                entries_as_label_score_pd = pd.DataFrame(
                    {
                        "label": entries["judge_name"],
                        "score": entries["judgement_score"].astype(float),
                    }
                )
                mean_score = jt.get_aggregation()(entries_as_label_score_pd)
                assert 0 <= mean_score <= 1
                formatted_scores[f"{prefix}/{group_name}"] = mean_score

            # Individual judge means
            for judge_name in sorted(all_judge_names):
                judge_entries = domain_entries[domain_entries["judge_name"] == judge_name]
                if len(judge_entries) == 0:
                    continue  # Judge not evaluated (e.g. ground_truth_similarity without answers)
                individual_score = float(np.mean(judge_entries["judgement_score"]))
                assert 0 <= individual_score <= 1
                formatted_scores[f"{prefix}/{judge_name}"] = individual_score

        return formatted_scores

    @beartype
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        domain_questions: dict[str, list[str]],
        n_max_openai_requests: int = 1_800,
        domain_answers: Optional[dict[str, list[str]]] = None,
    ) -> tuple[dict[str, float], str]:
        """
        Evaluate utility (biology) and safety/refusal (OOD domains).

        Args:
            model: HuggingFace model with .generate()
            tokenizer: HuggingFace tokenizer
            domain_questions: raw question strings per domain, e.g.
                {"biology": ["What is DNA?", ...], "cybersecurity": [...], ...}
            n_max_openai_requests: cost guard — raises if judge requests exceed this

        Returns:
            (scores_dict, df_as_json) where scores_dict has keys like
            "llm_judge/biology/in_scope/utility",
            "llm_judge/physics/out_of_scope/utility", etc.
        """
        if self.train_domain is None:
            assert all(d in _STATIC_DOMAIN_TO_SCOPE for d in domain_questions), (
                f"Unknown domain(s): {set(domain_questions) - set(_STATIC_DOMAIN_TO_SCOPE)}. "
                "Pass train_domain= to OneClickLLMJudgeScopingEval for dynamic scope."
            )

        # ── 1. Format prompts (user turn only, add_generation_prompt=True) ────
        prompt2seed: dict[str, str] = {}
        prompt2ground_truth: dict[str, str] = {}
        domain2prompts: dict[str, list[str]] = {}
        domain2sampled: dict[str, list[str]] = {}
        for domain, questions in domain_questions.items():
            answers = domain_answers.get(domain) if domain_answers is not None else None
            q2a: Optional[dict[str, str]] = None
            if answers is not None:
                assert len(answers) == len(questions), (
                    f"domain_answers length mismatch for {domain}: "
                    f"{len(answers)} answers vs {len(questions)} questions"
                )
                q2a = dict(zip(questions, answers))
            formatted = []
            sampled = random.Random(42).sample(questions, min(self.n_samples, len(questions)))
            domain2sampled[domain] = sampled
            for q in sampled:
                fp = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(fp)
                if fp not in prompt2seed:
                    prompt2seed[fp] = q
                if q2a is not None and fp not in prompt2ground_truth:
                    prompt2ground_truth[fp] = q2a[q]
            domain2prompts[domain] = formatted

        # ── 2. Build all_prompts = [(formatted_prompt, judge_name, domain), ...] ──────
        all_prompts: list[tuple[str, str, str]] = []
        for domain, fps in domain2prompts.items():
            for jt in DOMAIN_TO_JUDGE_TYPES.get(domain, _ALL_DOMAIN_JUDGES).values():
                for judge_name in jt.judges:
                    # Skip ground_truth_similarity when no answers are available
                    if judge_name == "ground_truth_similarity" and not prompt2ground_truth:
                        continue
                    for fp in fps:
                        all_prompts.append((fp, judge_name, domain))

        # ── 3. Cost guard ─────────────────────────────────────────────────────
        if len(all_prompts) > n_max_openai_requests:
            raise TooManyRequestsErrorLocal(
                f"Too many judge requests: {len(all_prompts)} > {n_max_openai_requests}"
            )
        if (
            self.n_max_openai_requests is not None
            and len(all_prompts) > self.n_max_openai_requests - self.n_requests
        ):
            raise TooManyRequestsErrorGlobal(
                f"Global limit exceeded: {len(all_prompts)} + {self.n_requests} "
                f"> {self.n_max_openai_requests}"
            )

        # ── 4. Run inference (unique prompts only, grouped by per-domain kwargs) ─
        prompt_to_domain: dict[str, str] = {
            fp: domain for domain, fps in domain2prompts.items() for fp in fps
        }
        unique_prompts = list(dict.fromkeys(fp for fp, _, _d in all_prompts))
        # Group by effective generation kwargs so different domains can use different settings.
        kwargs_to_prompts: dict[str, list[str]] = {}
        kwargs_lookup: dict[str, dict] = {}
        for fp in unique_prompts:
            domain = prompt_to_domain.get(fp, "")
            eff_kwargs = self.domain_generation_kwargs.get(domain, self.generation_kwargs)
            key = json.dumps(eff_kwargs, sort_keys=True)
            kwargs_to_prompts.setdefault(key, []).append(fp)
            kwargs_lookup[key] = eff_kwargs
        prompt2response: dict[str, str] = {}
        for key, group_prompts in kwargs_to_prompts.items():
            group_result = self._run_inference(model, tokenizer, group_prompts, generation_kwargs=kwargs_lookup[key])
            for idx, (_inp, out) in group_result.items():
                prompt2response[group_prompts[idx]] = out

        # ── 5. Run LLM judges ─────────────────────────────────────────────────
        df = self._run_llm_judges(
            all_prompts, prompt2seed, prompt2response,
            prompt2ground_truth=prompt2ground_truth if prompt2ground_truth else None,
        )

        # ── 6. Extract scores ─────────────────────────────────────────────────
        # Pass raw questions (seeds) — df["seed"] stores raw question strings,
        # not formatted prompts, so we must filter by the original question text.
        formatted_scores = self._extract_scores(df, domain2sampled)

        df_as_json: str = df.to_json(orient="records")
        return formatted_scores, df_as_json
