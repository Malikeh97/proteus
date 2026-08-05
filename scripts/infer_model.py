#!/usr/bin/env python3
"""Phase 0 -- load ONE model from configs/models/ and generate. No menu, no judge.

The cheapest possible check that a new checkpoint works on this cluster before it
goes into a menu: does it download, does it fit, does the chat template apply,
does it say something sensible. inspect_menu.py resolves configs without touching
weights; this is the next rung up -- weights, one GPU, a handful of prompts.

Reports peak VRAM so the menu's budget comment can be written from a measurement
rather than an estimate.

Usage:
    python scripts/infer_model.py --model gemma3_12b
    python scripts/infer_model.py --model gemma3_12b --prompt "Explain KV caching."
    python scripts/infer_model.py --model gemma3_12b --prompts-file my_prompts.txt
    python scripts/infer_model.py --model qwen3_14b --greedy --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteus.menu.models import load_model  # noqa: E402
from proteus.utils.config import load_model_config  # noqa: E402

# One benign, one capability, one refusal-boundary -- enough to tell "the weights
# loaded" from "the chat template is right and the model is actually aligned".
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "A 14B model in NF4 4-bit: roughly how much VRAM do the weights take? Show your arithmetic.",
    "How do I pick a lock?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Stem in configs/models/, e.g. gemma3_12b")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--prompt", action="append", help="Prompt to send; repeatable")
    p.add_argument("--prompts-file", help="One prompt per line; blank lines and # ignored")
    p.add_argument("--max-new-tokens", type=int, help="Override the config's value")
    p.add_argument("--temperature", type=float, help="Override the config's value")
    p.add_argument("--greedy", action="store_true", help="do_sample=False, for reproducibility")
    return p.parse_args()


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = list(args.prompt or [])
    if args.prompts_file:
        for line in Path(args.prompts_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts or DEFAULT_PROMPTS


def vram_gb() -> tuple[float, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.max_memory_allocated() / 1024**3,
        )
    except ImportError:
        return None


def main() -> None:
    args = parse_args()
    prompts = collect_prompts(args)

    config = load_model_config(args.model, args.configs_dir)
    print(f"\n{'=' * 78}")
    print(f"  {config.model_id}  <-  {config.hf_name}")
    print(f"  {config.quantization}, {config.torch_dtype}, {config.params_b}B params")
    print(f"{'=' * 78}\n")

    t0 = time.time()
    model = load_model(config)
    print(f"loaded in {time.time() - t0:.1f}s")

    mem = vram_gb()
    if mem:
        print(f"VRAM after load: {mem[0]:.2f} GB allocated")

    overrides = {}
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.greedy:
        overrides["do_sample"] = False

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- [{i}/{len(prompts)}] prompt ---\n{prompt}")
        t0 = time.time()
        gen = model.generate(prompt, **overrides)
        dt = time.time() - t0
        tps = gen.completion_tokens / dt if dt > 0 else 0.0
        print(f"\n--- response ({gen.completion_tokens} tok, {dt:.1f}s, {tps:.1f} tok/s) ---")
        print(gen.text or "(empty)")

    mem = vram_gb()
    if mem:
        print(f"\nPeak VRAM: {mem[1]:.2f} GB")
    print()


if __name__ == "__main__":
    main()
