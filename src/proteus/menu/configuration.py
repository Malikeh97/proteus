"""A configuration q = (M_i, S), S subset of W -- what actually serves one request."""

from __future__ import annotations

from dataclasses import dataclass, field

from proteus.menu.models.base import BaseModel
from proteus.menu.wrappers.base import Wrapper


@dataclass
class ServeResult:
    response: str
    refused: bool
    refused_by: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wrapper_tokens: int = 0
    fired: list[str] = field(default_factory=list)


def make_qid(model_id: str, wrapper_ids: list[str]) -> str:
    """Stable identity for a configuration. Bare model renders as just the model id."""
    if not wrapper_ids:
        return model_id
    return f"{model_id}+{'|'.join(sorted(wrapper_ids))}"


class Configuration:
    def __init__(self, model: BaseModel, wrappers: list[Wrapper] | None = None) -> None:
        self._model = model
        self._wrappers = list(wrappers or [])

    @property
    def qid(self) -> str:
        return make_qid(self._model.model_id, [w.wrapper_id for w in self._wrappers])

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def wrappers(self) -> list[Wrapper]:
        return list(self._wrappers)

    @property
    def wrapper_ids(self) -> list[str]:
        return [w.wrapper_id for w in self._wrappers]

    def serve(self, prompt: str, **kwargs) -> ServeResult:
        """Input wrappers -> model -> output wrappers. Any refusal short-circuits."""
        wrapper_tokens = 0
        fired: list[str] = []
        x = prompt

        for w in self._wrappers:
            if not w.runs_on_input:
                continue
            v = w.on_input(x)
            wrapper_tokens += v.tokens
            if not v.allow:
                fired.append(v.reason)
                return ServeResult(
                    response="",
                    refused=True,
                    refused_by=v.reason,
                    wrapper_tokens=wrapper_tokens,
                    fired=fired,
                )
            x = v.text

        gen = self._model.generate(x, **kwargs)
        y = gen.text

        for w in self._wrappers:
            if not w.runs_on_output:
                continue
            v = w.on_output(x, y)
            wrapper_tokens += v.tokens
            if not v.allow:
                fired.append(v.reason)
                return ServeResult(
                    response="",
                    refused=True,
                    refused_by=v.reason,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    wrapper_tokens=wrapper_tokens,
                    fired=fired,
                )
            y = v.text

        return ServeResult(
            response=y,
            refused=False,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            wrapper_tokens=wrapper_tokens,
            fired=fired,
        )

    def __repr__(self) -> str:
        return f"Configuration({self.qid})"
