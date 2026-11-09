from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from oppomem.prompts.rps import structured_text_parser as rps_structured_text_parser
from oppomem.prompts.ipd import structured_text_parser as ipd_structured_text_parser


def structured_text_parser(system):
    return rps_structured_text_parser(system) or ipd_structured_text_parser(system)


@dataclass(slots=True)
class LLMUsageEvent:
    stage: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_seconds: float
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_api_key(key_file: str | Path | None = None) -> str:
    """Load an API key without ever logging or returning its source path."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and key_file:
        path = Path(key_file)
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")
            key = next((line.strip() for line in text.splitlines()
                        if line.strip().startswith("sk-")), "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is unset and no usable key file was found")
    return key


def read_key_file(key_file: str | Path) -> str:
    """Read a key from a file, ignoring OPENAI_API_KEY.

    load_api_key() prefers the environment variable, which is the wrong behaviour for a
    SECOND provider in the same process: with OPENAI_API_KEY set, the agent's key would be
    silently sent to the opponent's base_url.
    """
    path = Path(key_file)
    if not path.is_file():
        raise RuntimeError(f"No key file at {path}")
    text = path.read_text(encoding="utf-8-sig")
    key = next((line.strip() for line in text.splitlines()
                if line.strip().startswith("sk-")), "")
    if not key:
        raise RuntimeError(f"No usable key line in {path}")
    return key


def load_base_url(key_file: str | Path | None = None) -> str | None:
    value = os.environ.get("OPENAI_BASE_URL", "").strip()
    if value:
        return value.rstrip("/")
    if key_file:
        path = Path(key_file)
        if path.exists():
            match = re.search(r"https?://[^\s]+", path.read_text(encoding="utf-8-sig"))
            if match:
                return match.group(0).strip().rstrip('\'"').rstrip("/")
    return None


def read_base_url_file(key_file: str | Path) -> str | None:
    """Read a base URL from a file, ignoring OPENAI_BASE_URL.

    The file-only counterpart of read_base_url, and it exists for the same reason
    read_key_file does: load_base_url() prefers the environment variable, so in a
    process running two providers it would hand BOTH clients whichever endpoint was
    exported.  A run that silently sends one provider's traffic to the other's host
    does not fail loudly -- it times out one call at a time through the retry
    ladder, which looks like a slow network rather than a misroute.
    """
    path = Path(key_file)
    if not path.is_file():
        return None
    match = re.search(r"https?://[^\s]+", path.read_text(encoding="utf-8-sig"))
    return match.group(0).strip().rstrip('\'"').rstrip("/") if match else None


# Ten attempts with an exponentially growing, jittered backoff, sized from real failures
# rather than guessed.  On 2026-09-17 the agent's proxy served 503s twice under load: a
# ~20 s window that stopped batch 1 at 235/360 cells, and a ~50 s window that killed 4 of
# the 8 batch-2 workers.  `benchmarks/runner.py` has no per-cell error tolerance, so one
# exhausted ladder kills the worker AND every cell still queued behind it.
#
# The 6-attempt ladder written after the first outage was not enough, and the reason is
# worth keeping.  Its jitter was symmetric, so a worker that drew low waited only
# 0.5*(1+2+4+8+16) = 15.5 s -- less than the outage it was meant to absorb.  That is
# exactly why the same 503 window killed 4 workers and spared the other 4: the survivors
# had drawn a long ladder.  The jitter below is biased upward instead (never shorter than
# nominal), and the nominal ladder alone is ~198 s.
#
# A call may now burn a few minutes before giving up.  That is the intended trade: waiting
# out a flaky window is far cheaper than losing the cell plus every cell behind it, since
# a restarted cell re-pays every agent call it had already made.
RETRY_ATTEMPTS = 10
RETRY_BACKOFF_CAP_SECONDS = 45.0
RETRY_JITTER_MIN = 0.8
RETRY_JITTER_MAX = 1.3


class OpenAIJsonLLM:
    """Responses API implementation of the small JSON-only core contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-mini",
        reasoning_effort: str = "minimal",
        max_attempts: int = RETRY_ATTEMPTS,
        request_limit: int | None = None,
        request_journal: str | Path | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        # An explicit base_url always wins.  OPENAI_BASE_URL is process-global and the
        # runner sets it to the AGENT's provider, so a second client in the same process
        # must never fall back to it.
        resolved = (base_url or "").strip() or (
            os.environ.get("OPENAI_BASE_URL", "").strip() or None
        )
        # SDK retries must not bypass the explicitly counted attempt budget.
        self.client = OpenAI(api_key=api_key, base_url=resolved, max_retries=0,
                             timeout=timeout)
        self.model = model
        # "default" (or None) means the PROVIDER decides: the reasoning parameter is
        # omitted from the request entirely rather than sent as a level.  The agent's own
        # calls want an explicit floor, but a reflection/segmenter judge asked to run at
        # its own default must not be forced down to "minimal" -- that would silently
        # make an expensive model behave like a cheap one.
        self.reasoning_effort = (
            None if reasoning_effort is None
            or str(reasoning_effort).strip().lower() in {"", "default", "none", "unset"}
            else str(reasoning_effort)
        )
        self.max_attempts = max(1, int(max_attempts))
        # Its own RNG instance, never the module-level one: benchmarks/overcooked.py seeds
        # the global `random` for reproducibility, and jitter here must not be able to
        # perturb anything a seed is supposed to pin down.
        self._jitter = random.Random()
        self.request_limit = None if request_limit is None else max(0, int(request_limit))
        self.requests_made = 0
        self.events: list[LLMUsageEvent] = []
        # Chars of hidden reasoning captured on the last transport call, for the journal.
        self._last_reasoning_chars = 0
        self.request_journal = Path(request_journal) if request_journal else None
        if self.request_journal and self.request_journal.is_file():
            self.requests_made = sum(
                json.loads(line).get("kind") == "request"
                for line in self.request_journal.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    def _journal(self, value: dict[str, Any]) -> None:
        if self.request_journal:
            self.request_journal.parent.mkdir(parents=True, exist_ok=True)
            with self.request_journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    @staticmethod
    def _usage(response: object) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        return input_tokens, output_tokens, reasoning_tokens

    # Providers whose hidden reasoning counts against the output budget need a floor here;
    # the subclass raises it.  complete_json's retry ladder doubles it on every failure.
    MIN_TOKEN_BUDGET = 512
    # ...and a ceiling, because at 10 attempts an uncapped doubling would climb from 512 to
    # 512 * 2**9 = 262144 and start drawing 400s that no retry can recover from -- turning
    # a transient outage into a permanent one.  Measured, not guessed: see
    # experiments/_code_archive/20260917_7961c13d/verification/_ds_maxtokens_probe.py
    MAX_TOKEN_BUDGET = 32768

    def _record_reasoning(self, *, stage: str, text: Any) -> None:
        """Journal a provider's hidden reasoning verbatim.

        Written from `_generate`, i.e. BEFORE the completion ladder parses the answer, so
        a call whose JSON arrives truncated -- the single most informative failure to
        read back -- still leaves its reasoning on disk.  Only providers that expose it
        call this; the responses API does not return its reasoning text.
        """
        value = str(text or "").strip()
        self._last_reasoning_chars = len(value)
        if value:
            self._journal({"kind": "reasoning", "attempt_id": self.requests_made,
                           "stage": stage, "text": value})

    def _generate(self, *, stage: str = "rps", system: str, prompt: str, token_budget: int,
                  text_mode: bool = False) -> tuple[str, tuple[int, int, int]]:
        """One transport call. Returns (text, (input, output, reasoning) tokens).

        Raises on anything unusable so the completion ladder can retry with a bigger
        budget.  Providers without a /responses endpoint override this.  ``text_mode``
        drops the JSON-object instruction and the json_object response format, which a
        natural-language stage (see ``core/tom_nl.py``) must not be given: the provider
        honours the format field literally and would refuse a plain sentence.
        """
        self._last_reasoning_chars = 0
        text_mode = text_mode or structured_text_parser(system) is not None
        if text_mode:
            response = self.client.responses.create(
                model=self.model,
                instructions=system,
                input=prompt,
                text={"verbosity": "low"},
                max_output_tokens=token_budget,
                store=False,
                **({"reasoning": {"effort": self.reasoning_effort}}
                   if self.reasoning_effort is not None else {}),
            )
        else:
            response = self.client.responses.create(
                model=self.model,
                instructions=system,
                input="Respond with one valid JSON object.\n" + prompt,
                text={"format": {"type": "json_object"}, "verbosity": "low"},
                max_output_tokens=token_budget,
                store=False,
                **({"reasoning": {"effort": self.reasoning_effort}}
                   if self.reasoning_effort is not None else {}),
            )
        status = str(getattr(response, "status", "completed"))
        if status != "completed":
            error = RuntimeError(f"Responses API status was {status!r}")
            error.response_status = status
            error.incomplete_reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            error.usage = self._usage(response) if getattr(response, "usage", None) is not None else None
            raise error
        return str(getattr(response, "output_text", "") or "").strip(), self._usage(response)

    @staticmethod
    def _parse_json_object(output_text: str) -> dict[str, Any]:
        # Some compatible providers wrap an otherwise valid JSON object in a fence.
        text = output_text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fenced:
            output_text = fenced.group(1)
        parsed = json.loads(output_text)
        if not isinstance(parsed, dict):
            raise ValueError("Model output was valid JSON but not an object")
        return parsed

    @staticmethod
    def _parse_text(output_text: str) -> str:
        text = str(output_text or "").strip()
        if not text:
            raise ValueError("Model returned empty text")
        return text

    def complete_json(
        self,
        *,
        stage: str,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        # Compatibility entry point: RPS models return text, locally wrapped as a dict.
        text_parser = structured_text_parser(system)
        return self._complete(stage=stage, system=system, prompt=prompt,
                              max_tokens=max_tokens, text_mode=text_parser is not None,
                              parse=text_parser or self._parse_json_object,
                              failure="JSON completion failed")

    def complete_text(
        self,
        *,
        stage: str,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Raw-text completion on the same ladder, journal, and request budget.

        Added for the natural-language ToM stage: unlike ``complete_json`` the result is
        a sentence, not an object, so nothing is parsed here beyond a non-empty check
        that keeps a truncated/empty answer on the retry ladder rather than committing
        an empty opponent read.
        """
        return self._complete(stage=stage, system=system, prompt=prompt,
                              max_tokens=max_tokens, text_mode=True,
                              parse=self._parse_text,
                              failure="Text completion failed")

    def _complete(
        self,
        *,
        stage: str,
        system: str,
        prompt: str,
        max_tokens: int | None,
        text_mode: bool,
        parse: Callable[[str], Any],
        failure: str,
    ) -> Any:
        started = time.perf_counter()
        totals = [0, 0, 0]
        last_error: Exception | None = None
        # Hidden reasoning counts against the output budget on both providers. Keep enough
        # headroom for minimal reasoning while retaining the core's visible limits.
        token_budget = max(self.MIN_TOKEN_BUDGET, int(max_tokens or self.MIN_TOKEN_BUDGET))

        for attempt in range(1, self.max_attempts + 1):
            try:
                if self.request_limit is not None and self.requests_made >= self.request_limit:
                    raise RuntimeError("API request limit exhausted")
                self.requests_made += 1
                self._journal({"kind": "request", "attempt_id": self.requests_made,
                               "stage": stage})
                output_text, usage = self._generate(
                    stage=stage, system=system, prompt=prompt,
                    token_budget=token_budget, text_mode=text_mode)
                self._journal({"kind": "response", "attempt_id": self.requests_made,
                               "stage": stage, "input_tokens": usage[0],
                               "output_tokens": usage[1], "reasoning_tokens": usage[2],
                               "reasoning_chars": self._last_reasoning_chars})
                totals = [a + b for a, b in zip(totals, usage)]
                parsed = parse(output_text)
                if stage == "memory_windows" and isinstance(parsed, dict):
                    self._journal({"kind": "summary_response", "attempt_id": self.requests_made,
                                   "stage": stage, "raw_response": parsed})
                self.events.append(LLMUsageEvent(
                    stage=stage,
                    input_tokens=totals[0],
                    output_tokens=totals[1],
                    reasoning_tokens=totals[2],
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                ))
                return parsed
            except Exception as exc:  # retry transport, truncation, and parse failures
                last_error = exc
                # Keep actionable diagnostics without persisting provider messages,
                # which may contain request content or credentials.
                status_code = getattr(exc, "status_code", None)
                self._journal({"kind": "error", "attempt_id": self.requests_made,
                               "stage": stage, "error_type": type(exc).__name__,
                               "http_status": status_code if isinstance(status_code, int) else None})
                if status_code in {400, 401, 402, 403, 404, 422}:
                    break  # Invalid credentials/model/parameters need a correction.
                if self.request_limit is not None and self.requests_made >= self.request_limit:
                    break
                token_budget = min(token_budget * 2, self.MAX_TOKEN_BUDGET)
                if attempt < self.max_attempts:
                    # Jittered so the 8 workers do not retry in lockstep and re-create the
                    # overload that produced the 503 -- but biased upward, never shorter
                    # than nominal, so no worker draws a ladder too short to outlast the
                    # outage it is waiting on.
                    delay = min(2.0 ** (attempt - 1), RETRY_BACKOFF_CAP_SECONDS)
                    time.sleep(delay * self._jitter.uniform(RETRY_JITTER_MIN,
                                                            RETRY_JITTER_MAX))

        self.events.append(LLMUsageEvent(
            stage=stage,
            input_tokens=totals[0],
            output_tokens=totals[1],
            reasoning_tokens=totals[2],
            latency_seconds=time.perf_counter() - started,
            attempts=attempt,
        ))
        status = getattr(last_error, "status_code", None)
        detail = type(last_error).__name__ + (f", HTTP {status}" if status else "")
        raise RuntimeError(f"{failure} at stage {stage!r} ({detail})") from last_error

    def usage_dicts(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


class DeepSeekJsonLLM(OpenAIJsonLLM):
    """Chat Completions transport, for providers with no /responses endpoint.

    Same contract, journal, request counting and retry ladder as OpenAIJsonLLM; only the
    transport differs, so the two providers' journals stay comparable.

    Two things about DeepSeek's flash model that this class is built around, both
    measured against the live API on 2026-09-17:

    * It is a REASONING model and `reasoning_tokens` bill against `max_tokens`. With a
      small budget it spends everything thinking and returns `content == ""` with
      `finish_reason == "length"` -- a silent empty answer, not an error. Hence
      MIN_TOKEN_BUDGET = 4000, and an explicit raise on empty content or a length stop so
      complete_json's ladder retries with double the budget instead of returning "".
    * `response_format={"type": "json_object"}` is supported; if a future deployment
      rejects it the call is retried once without it.
    """

    MIN_TOKEN_BUDGET = 4000

    def __init__(self, **kwargs: Any) -> None:
        effort = kwargs.get("reasoning_effort")
        super().__init__(**kwargs)
        # DeepSeek's documented "none" disables thinking; it must not be omitted.
        if effort == "none":
            self.reasoning_effort = "none"
        self._response_format_ok = True

    @staticmethod
    def _usage(response: object) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        return input_tokens, output_tokens, reasoning_tokens

    def _generate(self, *, stage: str = "rps", system: str, prompt: str, token_budget: int,
                  text_mode: bool = False) -> tuple[str, tuple[int, int, int]]:
        self._last_reasoning_chars = 0
        text_mode = text_mode or structured_text_parser(system) is not None
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt if text_mode
                 else "Respond with one valid JSON object.\n" + prompt},
            ],
            "max_tokens": token_budget,
        }
        # DeepSeek's current API documents none/low/high/max. Preserve an explicit
        # choice; the provider default is thinking enabled when the field is absent.
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.model == "qwen3.6-flash":
            payload["extra_body"] = {"enable_thinking": False}
        if self._response_format_ok and not text_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if self._response_format_ok and "response_format" in str(exc):
                self._response_format_ok = False
                payload.pop("response_format")
                response = self.client.chat.completions.create(**payload)
            else:
                raise
        if response is None or not getattr(response, "choices", None):
            raise RuntimeError("Chat Completions returned no choices")
        self._journal({"kind": "model_identity", "stage": stage, "requested_model": self.model, "returned_model": getattr(response, "model", None), "reasoning_effort": self.reasoning_effort})
        choice = response.choices[0]
        message = choice.message
        text = str(getattr(message, "content", "") or "").strip()
        # DeepSeek returns its thinking on the message's extra field, not in `content`.
        # Journaled here, before the empty-content raise below, so a truncated answer
        # still shows what the model was doing when the budget ran out.
        self._record_reasoning(
            stage=stage, text=getattr(message, "reasoning_content", None))
        if not text or getattr(choice, "finish_reason", None) == "length":
            error = RuntimeError(
                f"chat completion produced no usable content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r}) at a "
                f"{token_budget}-token budget; reasoning tokens count against it"
            )
            error.usage = self._usage(response) if getattr(response, "usage", None) is not None else None
            raise error
        return text, self._usage(response)
