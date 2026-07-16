"""HuggingFace transformers backend with bitsandbytes quantisation.

torch/transformers are imported inside methods so that CPU-only post-processing
(evaluate.py, plot_results.py) starts instantly and needs no GPU stack.
"""

from __future__ import annotations

from proteus.menu.models.base import BaseModel, Generation
from proteus.utils.config import ModelConfig
from proteus.utils.logging import get_logger

logger = get_logger("hf_model")

_BASE_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"


def build_quantization_config(quantization: str):
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class HFModel(BaseModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {config.model_id} ({config.hf_name}, {config.quantization})")

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.hf_tokenizer_id or config.hf_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            config.hf_name,
            quantization_config=build_quantization_config(config.quantization),
            torch_dtype=torch.float16,
            device_map=config.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _format(self, prompt: str) -> str:
        if self._config.model_type == "base" or self._tokenizer.chat_template is None:
            return _BASE_TEMPLATE.format(prompt=prompt)
        kwargs = {}
        if self._config.enable_thinking is not None:
            # Qwen3 templates accept this; others ignore unknown kwargs.
            kwargs["enable_thinking"] = self._config.enable_thinking
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompt: str, **kwargs) -> Generation:
        import torch

        gen = self._config.generation
        text = self._format(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        n_prompt = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
                temperature=kwargs.get("temperature", gen.temperature),
                top_p=kwargs.get("top_p", gen.top_p),
                do_sample=kwargs.get("do_sample", gen.do_sample),
                pad_token_id=self._tokenizer.pad_token_id,
            )

        completion_ids = out[0][n_prompt:]
        response = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return Generation(
            text=response,
            prompt_tokens=n_prompt,
            completion_tokens=int(completion_ids.shape[-1]),
        )
