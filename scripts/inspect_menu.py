#!/usr/bin/env python3
"""Phase 0 -- print the resource menu Q without loading any weights.

Cheap sanity check: confirms every stem reference resolves, shows |Q|, and lists
the configurations a coverage will range over. Run this before spending an
allocation on a menu that turns out to have 96 members.

Usage:
    python scripts/inspect_menu.py --menu full
    python scripts/inspect_menu.py --menu dev --verbose

    # What is registered?
    python scripts/inspect_menu.py --list-registries
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteus.menu.menu import ResourceMenu  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--menu", default="full", help="Stem in configs/menus/")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--verbose", action="store_true", help="Show hf_name and params per config")
    p.add_argument("--list-registries", action="store_true", help="List every registered component")
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print one model_id per line and exit -- the shard key run_probe.sh submits on",
    )
    return p.parse_args()


def list_registries() -> None:
    from proteus.attackers import ATTACKERS
    from proteus.benchmarks import BENCHMARKS
    from proteus.coverage.selectors import SELECTORS
    from proteus.judges import JUDGES
    from proteus.menu.models import MODELS
    from proteus.menu.wrappers import WRAPPERS

    for name, reg in [
        ("model backends", MODELS),
        ("wrappers", WRAPPERS),
        ("selectors", SELECTORS),
        ("attackers", ATTACKERS),
        ("judges", JUDGES),
        ("benchmarks", BENCHMARKS),
    ]:
        print(f"{name:16s} {', '.join(reg.names())}")


def main() -> None:
    args = parse_args()

    if args.list_registries:
        list_registries()
        return

    if args.list_models:
        # Machine-readable, nothing else on stdout: run_probe.sh reads this to fan
        # the probe out over models, so the launcher never duplicates the menu YAML.
        # ResourceMenu logs its shape to stdout on construction, so quieten it first.
        logging.getLogger("menu").setLevel(logging.WARNING)
        menu = ResourceMenu.from_name(args.menu, args.configs_dir)
        seen = dict.fromkeys(menu.describe(q)["model_id"] for q in menu.qids)
        print("\n".join(seen))
        return

    menu = ResourceMenu.from_name(args.menu, args.configs_dir)
    print(f"\nMenu '{menu.menu_id}': |Q| = {len(menu)}\n")

    for qid in menu.qids:
        d = menu.describe(qid)
        if args.verbose:
            print(f"  {qid}")
            print(f"      model    {d['hf_name']} ({d['params_b']}B)")
            print(f"      wrappers {', '.join(d['wrappers']) or '(none)'}")
        else:
            print(f"  {qid}")
    print()


if __name__ == "__main__":
    main()
