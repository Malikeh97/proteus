#!/usr/bin/env python3
"""Phase 4 -- plots. CPU only, runs on the login node.

Four figures, each answering one research question:
  frontier    safety-helpfulness frontier per selector      (RQ1)
  gap         static vs adaptive ASR per selector           (RQ2)
  entropy     coverage entropy and support size             (RQ3, falsification)
  cost        attacker FLOPs to first jailbreak             (RQ5)
  severity    harm severity of jailbreaks, by selector and configuration

Usage:
    python scripts/plot_results.py --metrics $PROTEUS_OUTPUT_DIR/analysis/metrics.csv
    python scripts/plot_results.py --metrics ... --figures frontier entropy

Writes:
    {metrics_dir}/plots/{figure}.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from proteus.utils.logging import get_logger  # noqa: E402

logger = get_logger("plot")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", required=True, help="Path to metrics.csv from evaluate.py")
    p.add_argument("--out-dir", help="Default: {metrics_dir}/plots")
    p.add_argument(
        "--figures",
        nargs="+",
        default=["frontier", "gap", "entropy", "cost", "severity"],
        choices=["frontier", "gap", "entropy", "cost", "severity"],
    )
    return p.parse_args()


def plot_frontier(df: pd.DataFrame, out: Path) -> None:
    """The object of comparison is the curve, not any single point."""
    last = df[df["round"] == df.groupby("selector")["round"].transform("max")]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for selector, g in last.groupby("selector"):
        g = g.groupby("tau", as_index=False).agg(
            asr=("asr", "mean"), helpfulness=("helpfulness", "mean")
        ).sort_values("helpfulness")
        ax.plot(g["helpfulness"], g["asr"], marker="o", label=selector)
    ax.set_xlabel("Helpfulness on benign traffic")
    ax.set_ylabel("Adaptive ASR")
    ax.set_title("Safety–helpfulness frontier")
    ax.legend(title="Selector")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_gap(df: pd.DataFrame, out: Path) -> None:
    """RQ2: randomization should narrow the gap between static and adaptive."""
    adaptive = {"pair", "pair-gemma", "grpo", "search"}
    d = df.copy()
    d["regime"] = d["attacker"].apply(lambda a: "adaptive" if a in adaptive else "static")
    g = d.groupby(["selector", "regime"], as_index=False).agg(asr=("asr", "mean"))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=g, x="selector", y="asr", hue="regime", ax=ax)
    ax.set_xlabel("Coverage selection strategy")
    ax.set_ylabel("ASR")
    ax.set_title("Static vs adaptive attack success")
    ax.legend(title="Attacker")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_entropy(df: pd.DataFrame, out: Path) -> None:
    """Whether c* mixes at all. Entropy 0 means the selector chose a router, and
    the framework's central claim does not apply."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    g = df.groupby("selector", as_index=False).agg(
        entropy=("entropy", "mean"), support=("support", "mean")
    )
    sns.barplot(data=g, x="selector", y="entropy", ax=axes[0])
    axes[0].set_ylabel("H(c) [nats]")
    axes[0].set_title("Coverage entropy")
    sns.barplot(data=g, x="selector", y="support", ax=axes[1])
    axes[1].set_ylabel("|supp(c)|")
    axes[1].set_title("Support size")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cost(df: pd.DataFrame, out: Path) -> None:
    """RQ5: the benefit is economic rather than absolute."""
    d = df[df["flops_to_first_jb"].notna() & (df["flops_to_first_jb"] > 0)]
    if d.empty:
        logger.warning("No successful jailbreaks recorded; skipping the cost figure")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    g = d.copy()
    g["tflops"] = g["flops_to_first_jb"] / 1e12
    sns.barplot(data=g, x="selector", y="tflops", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Coverage selection strategy")
    ax.set_ylabel("Attacker TFLOPs to first jailbreak")
    ax.set_title("Cost of a successful adaptive attack")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_severity(df: pd.DataFrame, out: Path, cfg_df: pd.DataFrame | None = None) -> None:
    """How *bad* the jailbreaks that get through are (0-10). ASR counts a hint and
    an execution-level blueprint identically; this separates them, per selection
    strategy and per configuration (which mechanism admits the worst responses)."""
    have_sel = "mean_jb_severity" in df.columns and df["mean_jb_severity"].notna().any()
    have_cfg = cfg_df is not None and not cfg_df.empty
    if not have_sel and not have_cfg:
        logger.warning("No jailbreak-severity data recorded; skipping the severity figure")
        return

    n = 2 if have_cfg else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    ax = axes[0][0]

    # Left: per-selector mean jailbreak severity.
    g = df.groupby("selector", as_index=False).agg(mean_jb_severity=("mean_jb_severity", "mean"))
    sns.barplot(data=g, x="selector", y="mean_jb_severity", ax=ax, color="#c0504d")
    ax.set_ylabel("Mean jailbreak severity (0-10)")
    ax.set_xlabel("")
    ax.set_ylim(0, 10)
    ax.set_title("Harm severity by selector")
    ax.tick_params(axis="x", rotation=30)

    # Right: per-configuration mean severity, jailbreak-count-weighted.
    if have_cfg:
        ax = axes[0][1]
        c = cfg_df.copy()
        c["weighted"] = c["mean_severity"] * c["n_jailbreaks"]
        agg = c.groupby("qid", as_index=False).agg(
            weighted=("weighted", "sum"), n=("n_jailbreaks", "sum")
        )
        agg["mean_severity"] = agg["weighted"] / agg["n"]
        agg = agg.sort_values("mean_severity")
        sns.barplot(data=agg, y="qid", x="mean_severity", ax=ax, color="#8064a2")
        ax.set_xlabel("Mean jailbreak severity (0-10)")
        ax.set_ylabel("")
        ax.set_xlim(0, 10)
        ax.set_title("Harm severity by configuration (mechanism)")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


_FIGURES = {
    "frontier": plot_frontier,
    "gap": plot_gap,
    "entropy": plot_entropy,
    "cost": plot_cost,
}


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    df = pd.read_csv(metrics_path)
    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-configuration severity, if evaluate.py wrote it next to metrics.csv.
    cfg_path = metrics_path.parent / "severity_by_config.csv"
    cfg_df = pd.read_csv(cfg_path) if cfg_path.exists() else None

    for name in args.figures:
        path = out_dir / f"{name}.pdf"
        if name == "severity":
            plot_severity(df, path, cfg_df)
        else:
            _FIGURES[name](df, path)
        # A figure may decline to render (no data); don't claim it was written.
        if path.exists():
            logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
