"""Plot the historical coefficient sweep, without selecting new parameters."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "c_summary.json").read_bytes())
    rows = list(csv.DictReader((args.run / "c_descriptive_grid.csv").open()))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for beta, label, color in [(0, "Max only + prior", "#889399"),
                               (.25, "25% joint evidence (current)", "#2375a5"),
                               (.7, "70% joint evidence (exploratory)", "#bc6b2f"),
                               (1, "Joint evidence only + prior", "#39917f")]:
        points = [(float(r["prior_weight"]), float(r["MAP@10"])) for r in rows
                  if float(r["beta"]) == beta]
        ax.plot(*zip(*points), marker="o", markersize=3, label=label, color=color)
    ax.scatter([4], [summary["baseline"]["MAP@10"]], color="#10283a", s=55, zorder=5)
    ax.annotate("Current: beta=0.25, lambda=4", (4, summary["baseline"]["MAP@10"]),
                xytext=(.53, .18), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#10283a"}, fontsize=9)
    ax.set_xscale("symlog", linthresh=1)
    ticks = [0, .5, 1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.set_xlabel("Initial-score weight (lambda; nonlinear axis after 1)")
    ax.set_ylabel("MAP@10")
    ax.set_title("C coefficient sensitivity: historical SWE 500", loc="left", fontweight="bold")
    ax.grid(alpha=.18)
    ax.legend(loc="lower left", fontsize=8)
    fig.text(.09, .015, "Fixed new-AB pool and original C top 10. Descriptive sweep, not GHUB/GHRB validation.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.run / f"c_coefficient_sensitivity.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
