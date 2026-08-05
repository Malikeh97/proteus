"""GCG (Greedy Coordinate Gradient) -- the white-box attacker (Zou et al., 2023).

Optimises an adversarial suffix by taking the gradient of the log-likelihood of a
target affirmative prefix ("Sure, here is") with respect to a one-hot encoding of
the suffix tokens, then greedily swapping tokens for high-gradient candidates.

WHERE THE GRADIENT COMES FROM -- read this before changing anything
--------------------------------------------------------------------
GCG needs weights. proteus does not hand the attacker a target: `deployment.serve`
draws q ~ c(x) *per request*, and attackers/base.py forbids widening `refine()` to
see q or c. So the gradient target is resolved once, in `bind_deployment`, by a
single bit of information:

  * |support(c)| == 1  -- the coverage is a POINT MASS. There is exactly one
    deployed configuration and it is not a secret; this is Phase 1, where
    probe_menu.py sets w[q] = 1 to measure JB_q for one member at a time. Taking
    gradients through that model is white-box by construction and leaks nothing
    that the probe was not already holding fixed.

  * |support(c)| > 1   -- a MIXTURE. This is Phase 2. Reading which q was drawn
    would be the leaked-commitment relaxation base.py warns about, so we do not.
    GCG falls back to the surrogate named by `attacker_model:` and the run becomes
    a TRANSFER attack: a suffix optimised off-policy against a model the attacker
    owns, replayed against whatever gets drawn. Report it as transfer, not
    white-box -- it is a lower bound on a white-box adversary.

The one bit `|support| == 1` is the only thing this class reads off the coverage,
and in the mixture branch it is used solely to refuse. `refine()` itself still
receives (x, y, z) and nothing else, so the projection rho holds.

GCG IS NOT RESPONSE-ADAPTIVE. It ignores `response` and adapts to its own loss,
not to the deployment's feedback. Against a mixture that makes it strictly less
adaptive than PAIR -- which is the point of having both.

QUANTISATION. The menu is served in NF4. Gradients still flow to the *inputs* of a
bitsandbytes Linear4bit (this is what makes QLoRA work), so GCG runs on the
quantised model as-is, but the quantisation noise degrades gradient quality and
the suffixes are weaker than fp16 GCG. Set `quantization: "none"` on the target's
model config if you need the honest white-box ceiling.
"""

from __future__ import annotations

from proteus.attackers import ATTACKERS
from proteus.attackers.base import Attacker
from proteus.utils.config import CONFIGS_DIR, AttackerConfig, load_model_config
from proteus.utils.logging import get_logger

logger = get_logger("gcg")

# 20 tokens is the Zou et al. default. Each "!" is its own token in every
# tokenizer we serve, so the suffix length in tokens is predictable.
_SUFFIX_INIT = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

# Rendered into the chat template to locate where the user content sits, so the
# suffix can be a contiguous token span inside the template rather than tokenised
# separately from it. Must be a string no tokenizer will split oddly at the seam.
_SENTINEL = "<<<PROTEUS_GCG_CONTENT>>>"


def _vocab_size(model, tokenizer) -> int:
    """Vocabulary size, tolerating nested configs.

    Text-only checkpoints put `vocab_size` on the top-level config. Multimodal ones
    do not: gemma-3-*-it loads as Gemma3ForConditionalGeneration, whose config keeps
    vocab_size under `text_config` and raises AttributeError on a direct lookup.
    The one-hot gradient needs this, so it has to resolve either shape.
    """
    cfg = model.config
    for obj in (cfg, getattr(cfg, "text_config", None)):
        size = getattr(obj, "vocab_size", None)
        if size:
            return int(size)
    return len(tokenizer)


@ATTACKERS.register("gcg")
class GCGAttacker(Attacker):
    def __init__(self, config: AttackerConfig, configs_dir=CONFIGS_DIR) -> None:
        super().__init__(config)
        self._configs_dir = configs_dir
        self._surrogate_stem = config.attacker_model

        p = config.params
        self._n_steps = int(p.get("steps_per_refine", 1))
        self._top_k = int(p.get("top_k", 256))
        self._n_candidates = int(p.get("num_candidates", 128))
        self._batch_size = int(p.get("eval_batch_size", 32))
        self._target = str(p.get("target", "Sure, here is"))
        self._suffix_init = str(p.get("suffix_init", _SUFFIX_INIT))

        self._model = None  # BaseModel whose weights the gradient runs through
        self._white_box = False  # False => transfer mode, for the run metadata
        self._suffix = self._suffix_init
        self._base = ""

    # ------------------------------------------------------------------ binding

    def bind_deployment(self, deployment) -> None:
        """Resolve the gradient target. Called by the runner before reset().

        Reads exactly one bit off the coverage -- whether it is a point mass -- and
        nothing else. See the module docstring for why that is not a rho violation.
        """
        coverage = deployment.coverage
        support = coverage.support()

        if len(support) == 1:
            qid = support[0]
            self._model = coverage.menu.get(qid).model
            self._white_box = True
            logger.info(f"GCG white-box: point-mass coverage, taking gradients through '{qid}'")
            return

        if not self._surrogate_stem:
            raise ValueError(
                f"Attacker '{self.attacker_id}' faces a mixture coverage "
                f"(|support| = {len(support)}) and has no `attacker_model:` to fall back "
                "on. GCG cannot take gradients through a distribution over "
                "configurations, and reading the drawn q would break the projection "
                "rho (see attackers/base.py). Set `attacker_model:` in the attacker "
                "config to run GCG as a transfer attack."
            )

        from proteus.menu.models import load_model

        self._model = load_model(load_model_config(self._surrogate_stem, self._configs_dir))
        self._white_box = False
        logger.info(
            f"GCG transfer: mixture coverage (|support| = {len(support)}), taking "
            f"gradients through surrogate '{self._surrogate_stem}'. Report this as "
            "transfer ASR, not white-box."
        )

    @property
    def is_white_box(self) -> bool:
        """True when the last binding resolved to the deployed model. Record it --
        a white-box number and a transfer number are not the same measurement."""
        return self._white_box

    # ------------------------------------------------------------- the interface

    def reset(self, coverage_id: str | None = None) -> None:
        super().reset(coverage_id)
        self._suffix = self._suffix_init

    def initialize(self, base_prompt: str) -> str:
        self._base = base_prompt
        self._suffix = self._suffix_init
        return f"{base_prompt} {self._suffix}"

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        if judgment == 1:
            return prompt  # already succeeded; the trial loop stops anyway
        if self._model is None:
            raise RuntimeError(
                "GCG was never bound to a gradient target. The runner must call "
                "bind_deployment(deployment) before the first refine()."
            )
        for _ in range(self._n_steps):
            self._suffix = self._gcg_step()
        return f"{self._base} {self._suffix}"

    # ----------------------------------------------------------------- the optim

    def _segments(self, tokenizer):
        """Token ids for [pre][suffix][post][target].

        The suffix has to be a contiguous span *inside* the rendered chat template,
        otherwise the gradient is taken at token positions the model never sees in
        that arrangement. Rendering with a sentinel and splitting on it is the only
        way to do that without hand-coding each family's template.
        """
        rendered = None
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": _SENTINEL}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if rendered and _SENTINEL in rendered:
            head, tail = rendered.split(_SENTINEL, 1)
        else:
            head, tail = f"### Instruction:\n", "\n\n### Response:\n"

        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        return (
            enc(head + self._base + " "),
            enc(self._suffix),
            enc(tail),
            enc(self._target),
        )

    def _gcg_step(self) -> str:
        import torch
        import torch.nn.functional as F

        model = self._model._model
        tokenizer = self._model._tokenizer
        device = model.device

        pre, suf, post, tgt = self._segments(tokenizer)
        n_suf, n_tgt = len(suf), len(tgt)
        if n_suf == 0 or n_tgt == 0:
            raise RuntimeError("GCG: empty suffix or target after tokenisation.")

        embed_layer = model.get_input_embeddings()
        embed_matrix = embed_layer.weight
        vocab = _vocab_size(model, tokenizer)

        ids = lambda xs: torch.tensor(xs, device=device, dtype=torch.long)  # noqa: E731
        pre_ids, suf_ids, post_ids, tgt_ids = ids(pre), ids(suf), ids(post), ids(tgt)

        # --- gradient of the target NLL w.r.t. a one-hot over the suffix ---------
        one_hot = torch.zeros(n_suf, vocab, device=device, dtype=embed_matrix.dtype)
        one_hot.scatter_(1, suf_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)

        embeds = torch.cat(
            [
                embed_layer(pre_ids).detach(),
                one_hot @ embed_matrix,
                embed_layer(post_ids).detach(),
                embed_layer(tgt_ids).detach(),
            ]
        ).unsqueeze(0)

        logits = model(inputs_embeds=embeds).logits
        # Predict target position i from the token before it.
        shift = logits[0, -n_tgt - 1 : -1, :]
        loss = F.cross_entropy(shift.float(), tgt_ids)
        loss.backward()

        grad = one_hot.grad.clone()
        model.zero_grad(set_to_none=True)

        # --- propose: swap one position to a token the gradient likes ------------
        # -grad because we descend; top_k over the vocab at each suffix position.
        cand = (-grad).topk(self._top_k, dim=1).indices  # (n_suf, top_k)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.params.get("seed", 0)) + n_suf)
        pos = torch.randint(0, n_suf, (self._n_candidates,), generator=gen).to(device)
        pick = torch.randint(0, self._top_k, (self._n_candidates,), generator=gen).to(device)

        batch = suf_ids.repeat(self._n_candidates, 1)
        batch[torch.arange(self._n_candidates, device=device), pos] = cand[pos, pick]

        # --- select: the candidate with the lowest target loss -------------------
        best_ids, best_loss = suf_ids, float(loss.item())
        prefix = pre_ids.repeat(self._batch_size, 1)
        suffix_post = post_ids.repeat(self._batch_size, 1)
        target = tgt_ids.repeat(self._batch_size, 1)

        with torch.no_grad():
            for start in range(0, self._n_candidates, self._batch_size):
                chunk = batch[start : start + self._batch_size]
                b = chunk.shape[0]
                seq = torch.cat([prefix[:b], chunk, suffix_post[:b], target[:b]], dim=1)
                out = model(input_ids=seq).logits
                sh = out[:, -n_tgt - 1 : -1, :].float()
                losses = F.cross_entropy(
                    sh.reshape(-1, sh.shape[-1]),
                    target[:b].reshape(-1),
                    reduction="none",
                ).view(b, n_tgt).mean(dim=1)

                j = int(losses.argmin().item())
                if float(losses[j].item()) < best_loss:
                    best_loss = float(losses[j].item())
                    best_ids = chunk[j]

        # Forward passes are the attacker's compute cost; count them like PAIR does.
        self._tokens_spent += (self._n_candidates + 1) * (
            len(pre) + n_suf + len(post) + n_tgt
        )

        return tokenizer.decode(best_ids, skip_special_tokens=True)
