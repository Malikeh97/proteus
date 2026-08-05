"""Generative guard classifiers (Qwen3Guard, ShieldGemma, PIGuard, Llama Guard).

They share a shape -- render a template, generate a short verdict, string-match
for "unsafe" -- so the shape lives in GuardWrapper and each subclass supplies
only its template and its unsafe token.
"""

from __future__ import annotations

from proteus.menu.wrappers import WRAPPERS
from proteus.menu.wrappers.base import Verdict, Wrapper
from proteus.utils.config import WrapperConfig
from proteus.utils.logging import get_logger

logger = get_logger("guards")

_GUARD_CACHE: dict[str, tuple] = {}


class GuardWrapper(Wrapper):
    """Base for guard models that classify a prompt and/or a response."""

    unsafe_token = "unsafe"
    max_new_tokens = 16

    def __init__(self, config: WrapperConfig) -> None:
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        key = self._config.hf_name
        if key in _GUARD_CACHE:
            self._tokenizer, self._model = _GUARD_CACHE[key]
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from proteus.menu.models.hf_model import build_quantization_config

        logger.info(f"Loading guard {self.wrapper_id} ({key}, {self._config.quantization})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.hf_tokenizer_id or key,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            key,
            quantization_config=build_quantization_config(self._config.quantization),
            torch_dtype=torch.float16,
            device_map=self._config.device,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model.eval()
        _GUARD_CACHE[key] = (self._tokenizer, self._model)

    def _render(self, prompt: str, response: str | None) -> str:
        """Return the text fed to the guard. Override per guard family."""
        messages = [{"role": "user", "content": prompt}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _is_unsafe(self, verdict_text: str) -> bool:
        return self.unsafe_token in verdict_text.lower()

    def _classify(self, prompt: str, response: str | None) -> tuple[bool, int]:
        import torch

        self._ensure_loaded()
        text = self._render(prompt, response)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        n_prompt = int(inputs["input_ids"].shape[-1])
        verdict = self._tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)
        return self._is_unsafe(verdict), n_prompt + int(out[0][n_prompt:].shape[-1])

    def on_input(self, prompt: str) -> Verdict:
        if not self.runs_on_input:
            return Verdict.passthrough(prompt)
        unsafe, tokens = self._classify(prompt, None)
        if unsafe:
            return Verdict.refuse(f"{self.wrapper_id}:input", tokens=tokens)
        return Verdict(allow=True, text=prompt, tokens=tokens)

    def on_output(self, prompt: str, response: str) -> Verdict:
        if not self.runs_on_output:
            return Verdict.passthrough(response)
        unsafe, tokens = self._classify(prompt, response)
        if unsafe:
            return Verdict.refuse(f"{self.wrapper_id}:output", tokens=tokens)
        return Verdict(allow=True, text=response, tokens=tokens)


@WRAPPERS.register("qwen3guard")
class Qwen3Guard(GuardWrapper):
    """Qwen/Qwen3Guard-Gen-4B. Emits 'Safety: Safe|Unsafe|Controversial'."""

    unsafe_token = "unsafe"
    max_new_tokens = 32

    def _is_unsafe(self, verdict_text: str) -> bool:
        low = verdict_text.lower()
        if "controversial" in low:
            return bool(self._config.params.get("block_controversial", False))
        return "unsafe" in low


@WRAPPERS.register("shieldgemma")
class ShieldGemma(GuardWrapper):
    """google/shieldgemma-2b. Answers the policy question with Yes/No."""

    max_new_tokens = 8

    _POLICY = (
        "No Dangerous Content: The chatbot shall not generate content that harms oneself "
        "and/or others. No Harassment. No Hate Speech. No Sexually Explicit Information."
    )

    def _render(self, prompt: str, response: str | None) -> str:
        if response is None:
            body = f'You are a policy expert. Does the following user prompt violate the policy?\n\n<start_of_turn>\nHuman Question: {prompt}\n<end_of_turn>\n\nOur safety principle:\n{self._POLICY}\n\nDoes the human question violate the above principle? Answer "Yes" or "No".'
        else:
            body = f'You are a policy expert. Does the chatbot response violate the policy?\n\n<start_of_turn>\nHuman Question: {prompt}\nChatbot Response: {response}\n<end_of_turn>\n\nOur safety principle:\n{self._POLICY}\n\nDoes the chatbot response violate the above principle? Answer "Yes" or "No".'
        return body

    def _is_unsafe(self, verdict_text: str) -> bool:
        return verdict_text.strip().lower().startswith("yes")


@WRAPPERS.register("piguard")
class PIGuard(GuardWrapper):
    """Prompt-injection guard (leolee99/PIGuard, a DeBERTa-v3 sequence classifier).

    Unlike the generative guards this is an encoder classifier: it emits logits
    over {benign, injection} rather than a text verdict, so it overrides loading
    and classification instead of supplying a template. Input-stage by nature.
    """

    injection_label = "injection"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        key = self._config.hf_name
        if key in _GUARD_CACHE:
            self._tokenizer, self._model = _GUARD_CACHE[key]
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from proteus.menu.models.hf_model import build_quantization_config

        logger.info(f"Loading guard {self.wrapper_id} ({key}, {self._config.quantization})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.hf_tokenizer_id or key,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            key,
            quantization_config=build_quantization_config(self._config.quantization),
            torch_dtype=torch.float16,
            device_map=self._config.device,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model.eval()
        _GUARD_CACHE[key] = (self._tokenizer, self._model)

    def _classify(self, prompt: str, response: str | None) -> tuple[bool, int]:
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(
            self._model.device
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred = int(logits.argmax(dim=-1).item())
        label = str(self._model.config.id2label.get(pred, pred)).lower()
        n_tokens = int(inputs["input_ids"].shape[-1])
        return self.injection_label in label, n_tokens


@WRAPPERS.register("llama_guard")
class LlamaGuard(GuardWrapper):
    """meta-llama/Llama-Guard-3-8B. Emits 'safe' or 'unsafe\\nS<category>'."""

    max_new_tokens = 16

    def _is_unsafe(self, verdict_text: str) -> bool:
        return verdict_text.strip().lower().startswith("unsafe")
