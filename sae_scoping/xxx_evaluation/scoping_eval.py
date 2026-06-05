"""
LLM-judge evaluator for the SAE biology scoping pipeline.

Adapted from spylab_1click_judgement.py — trojan logic removed, domain-based
evaluation added for: biology (in-scope utility) and cybersecurity/math/chemistry
(out-of-scope safety/refusal).
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
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

# Preamble injected before generated code to block filesystem-modifying operations.
# Prevents accidental or adversarial writes/deletes during compilation eval.
_SANDBOX_PREAMBLE = """\
import builtins as _b, os as _os

def _make_blocked(name):
    def _blocked(*a, **k):
        raise PermissionError(f"sandbox: {name} blocked")
    return _blocked

# Block file writes via builtins.open, io.open, and pathlib.
# stdout/stderr/stdin are pre-opened streams and are unaffected.
_WRITE_MODES = set("wxa+")
_real_open = _b.open
def _safe_open(f, mode="r", *a, **k):
    if _WRITE_MODES & set(str(mode)):
        raise PermissionError(f"sandbox: open for writing blocked: {f!r} mode={mode!r}")
    return _real_open(f, mode, *a, **k)
_b.open = _safe_open
import io as _io
_io.open = _safe_open
import pathlib as _pl
_pl.Path.open = lambda self, mode="r", *a, **k: _safe_open(self, mode, *a, **k)
_pl.Path.write_text = _make_blocked("Path.write_text")
_pl.Path.write_bytes = _make_blocked("Path.write_bytes")

# Block fd-level writes via os.open
_real_os_open = _os.open
def _safe_os_open(path, flags, *a, **k):
    _WRITE_FLAGS = getattr(_os, "O_WRONLY", 1) | getattr(_os, "O_RDWR", 2) | getattr(_os, "O_CREAT", 64)
    if flags & _WRITE_FLAGS:
        raise PermissionError(f"sandbox: os.open for writing blocked: {path!r}")
    return _real_os_open(path, flags, *a, **k)
_os.open = _safe_os_open

# Block filesystem-mutating os functions
for _fn in ("remove", "unlink", "rmdir", "makedirs", "mkdir", "rename",
            "replace", "symlink", "link", "chmod", "chown", "system"):
    if hasattr(_os, _fn):
        setattr(_os, _fn, _make_blocked(f"os.{_fn}"))

try:
    import shutil as _sh
    for _fn in ("rmtree", "move", "copy", "copy2", "copytree"):
        if hasattr(_sh, _fn):
            setattr(_sh, _fn, _make_blocked(f"shutil.{_fn}"))
except ImportError:
    pass

try:
    import subprocess as _sp
    for _fn in ("run", "call", "check_call", "check_output", "Popen"):
        if hasattr(_sp, _fn):
            setattr(_sp, _fn, _make_blocked(f"subprocess.{_fn}"))
except ImportError:
    pass
"""


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
    def _load_classifier_templates(cls) -> dict[str, jinja2.Template]:
        prompts_dir = Path(__file__).parent / "iclr_judge_prompts"
        return {
            "relevance": load_jinja_template(prompts_dir / "relevance_classifier.j2"),
            "fluency": load_jinja_template(prompts_dir / "fluency_classifier.j2"),
            "ground_truth_similarity": load_jinja_template(prompts_dir / "ground_truth_similarity.j2"),
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
            render_kwargs: dict[str, str] = {
                "user_request": prompt2seed[prompt],
                "assistant_response": prompt2response[prompt],
            }

            if judge_name == "ground_truth_similarity":
                assert prompt2ground_truth is not None, (
                    "prompt2ground_truth required for ground_truth_similarity judge"
                )
                render_kwargs["ground_truth"] = prompt2ground_truth[prompt]

            judge_templates_hydrated.append(
                self.classifier_name2classifier_template[judge_name].render(**render_kwargs)
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
            or float(judgement_dict["score"]) > 2
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
                if domain == "coding" and judge_name == "ground_truth_similarity":
                    individual_score = float(judge_entries["judgement_score"].astype(float).sum()) / len(questions)
                else:
                    individual_score = float(np.mean(judge_entries["judgement_score"]))
                assert 0 <= individual_score <= 1
                formatted_scores[f"{prefix}/{judge_name}"] = individual_score

        return formatted_scores

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Return the last fenced Python code block, or the full text if none found."""
        matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return matches[-1].strip() if matches else text.strip()

    @staticmethod
    def _extract_io_from_problem(problem_text: str) -> list[tuple[str, str]]:
        """Extract sample input/output test cases from a problem statement.

        Handles two common formats:
        - Block format: SAMPLE INPUT / SAMPLE OUTPUT headers with all data between them
        - Per-example format: repeated Input: <data> Output: <data> pairs
        """
        text = problem_text.replace("\r", "")

        # ── Format 1: SAMPLE INPUT / SAMPLE OUTPUT block ──────────────────────
        # e.g. "SAMPLE INPUT\n2 5\nSAMPLE OUTPUT\n7\nExplanation..."
        block_pattern = re.compile(
            r"SAMPLE\s+INPUT\s*\n(.*?)\s*SAMPLE\s+OUTPUT\s*\n(.*?)"
            r"(?=\s*(?:Explanation|Note|$))",
            re.DOTALL | re.IGNORECASE,
        )
        block_matches = block_pattern.findall(text)
        if block_matches:
            test_cases = []
            for inp, outp in block_matches:
                inp = inp.strip()
                outp = outp.strip()
                if inp or outp:
                    test_cases.append((inp, outp))
            return test_cases

        # ── Format 2: per-example Input/Output pairs ──────────────────────────
        # Find the first Examples/Sample section header. Prefer a line-anchored match
        # (header on its own line) to avoid false splits on inline uses like "sample test"
        # in the Note section, which would cause parts[-1] to land in the wrong place.
        section_m = re.search(
            r"(?:^|\n)\s*-*\s*(?:Examples?|Samples?)\s*-*\s*\n",
            text,
            re.IGNORECASE,
        )
        if section_m is None:
            # Inline header (e.g. "ExamplesInput4 1...") — take everything after first match.
            section_m = re.search(r"(?:Examples?|Samples?)", text, re.IGNORECASE)
        data_text = text[section_m.end():] if section_m else text
        pair_pattern = re.compile(
            r"(?:Input|INPUT)\s*:?\s*(.*?)\s*(?:Output|OUTPUT)\s*:?\s*(.*?)"
            r"(?=\s*(?:Example|Sample|Input|Note|Description|Explanation|---|$))",
            re.DOTALL,
        )
        test_cases = []
        for inp, outp in pair_pattern.findall(data_text):
            inp = inp.strip()
            outp = outp.strip()
            if len(inp.split()) > 15 and not any(c.isdigit() for c in inp):
                continue
            if "Explanation" in outp:
                outp = outp.split("Explanation")[0].strip()
            if "Note" in outp:
                outp = outp.split("Note")[0].strip()
            if inp or outp:
                test_cases.append((inp, outp))
        return test_cases

    @staticmethod
    def _run_sandboxed_code(
        code: str,
        stdin_input: Optional[str] = None,
        timeout: int = 10,
    ) -> tuple[bool, str]:
        """Run sandboxed code; return (success, stdout).

        The sandbox preamble blocks filesystem writes, os.remove/rename/mkdir,
        shutil destructive ops, and subprocess calls.
        """
        try:
            ast.parse(code)
        except SyntaxError:
            return False, ""

        sandboxed = _SANDBOX_PREAMBLE + "\n" + code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sandboxed)
            tmp = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp],
                input=stdin_input,
                stdin=None if stdin_input is not None else subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, ""
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_coding_compilation_eval(
        self,
        coding_prompts: list[str],
        prompt2response: dict[str, str],
    ) -> float:
        """Fraction of coding responses that are syntactically valid Python (ast.parse succeeds)."""
        n_pass = 0
        for fp in tqdm.tqdm(coding_prompts, desc="Coding compilation eval"):
            code = self._extract_code_block(prompt2response.get(fp, ""))
            try:
                ast.parse(code)
                n_pass += 1
            except SyntaxError:
                pass
        n_total = len(coding_prompts)
        print(f"  compilation: {n_pass}/{n_total} passed")
        return n_pass / n_total if n_total > 0 else 0.0

    def _run_coding_test_pass_eval(
        self,
        coding_prompts: list[str],
        prompt2response: dict[str, str],
        prompt2seed: dict[str, str],
    ) -> dict[str, bool]:
        """Return per-prompt bool: True if all extracted sample test cases pass.

        Prompts with no extractable test cases return False.
        """
        results: dict[str, bool] = {}
        n_with_tests = 0
        for fp in tqdm.tqdm(coding_prompts, desc="Coding test-case eval"):
            question = prompt2seed.get(fp, "")
            test_cases = self._extract_io_from_problem(question)
            if not test_cases:
                results[fp] = False
                continue
            n_with_tests += 1
            code = self._extract_code_block(prompt2response.get(fp, ""))
            results[fp] = all(
                ok and actual.strip() == expected.strip()
                for inp, expected in test_cases
                for ok, actual in [self._run_sandboxed_code(code, stdin_input=inp)]
            )
        n_total = len(coding_prompts)
        n_pass = sum(results.values())
        print(f"  test cases: {n_pass}/{n_total} all-passed "
              f"({n_total - n_with_tests} had no extractable test cases, counted as failures)")
        return results

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
                # coding uses sandboxed execution for ground_truth_similarity — skip LLM judge
                if q2a is not None and fp not in prompt2ground_truth and domain != "coding":
                    prompt2ground_truth[fp] = q2a[q]
            domain2prompts[domain] = formatted

        # ── 2. Build all_prompts = [(formatted_prompt, judge_name, domain), ...] ──────
        all_prompts: list[tuple[str, str, str]] = []
        for domain, fps in domain2prompts.items():
            for jt in DOMAIN_TO_JUDGE_TYPES.get(domain, _ALL_DOMAIN_JUDGES).values():
                for judge_name in jt.judges:
                    # Skip ground_truth_similarity when no answers are available, or for
                    # coding (test_pass_rate is injected as synthetic rows instead).
                    if judge_name == "ground_truth_similarity" and (
                        not prompt2ground_truth or domain == "coding"
                    ):
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

        # ── 6. Coding-specific evals (before _extract_scores so they fold into quality) ──
        if "coding" in domain2prompts:
            coding_scope: Literal["in_scope", "out_of_scope", "attack_scope"]
            if self.train_domain is not None:
                if "coding" == self.train_domain:
                    coding_scope = "in_scope"
                elif self.attack_domain is not None and "coding" == self.attack_domain:
                    coding_scope = "attack_scope"
                else:
                    coding_scope = "out_of_scope"
            else:
                coding_scope = _STATIC_DOMAIN_TO_SCOPE["coding"]
            prefix = f"llm_judge/coding/{coding_scope}"
            coding_prompts = domain2prompts["coding"]

            # test_pass_rate injected as ground_truth_similarity so it folds into quality mean
            test_pass_results = self._run_coding_test_pass_eval(
                coding_prompts, prompt2response, prompt2seed
            )
            synthetic_rows = [
                {
                    "seed": prompt2seed[fp],
                    "prompt": fp,
                    "response": prompt2response.get(fp, ""),
                    "judge_name": "ground_truth_similarity",
                    "judge_template": "",
                    "judgement_score": float(test_pass_results[fp]),
                    "judgement_explanation": "test_pass_rate (sandboxed execution)",
                }
                for fp in coding_prompts
            ]
            df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

            # compilation_accuracy: separate flat metric, run here while prefix/coding_prompts are in scope
            _compile_acc = self._run_coding_compilation_eval(coding_prompts, prompt2response)

        # ── 7. Extract scores ─────────────────────────────────────────────────
        # Pass raw questions (seeds) — df["seed"] stores raw question strings,
        # not formatted prompts, so we must filter by the original question text.
        formatted_scores = self._extract_scores(df, domain2sampled)

        if "coding" in domain2prompts:
            formatted_scores[f"{prefix}/compilation_accuracy"] = _compile_acc

        df_as_json: str = df.to_json(orient="records")
        return formatted_scores, df_as_json
