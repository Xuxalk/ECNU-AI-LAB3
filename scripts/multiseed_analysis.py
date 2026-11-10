import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

runs = {
    "B": {
        42: "cnn_crop_flip_v1",
        123: "cnn_crop_flip_seed123",
        2026: "cnn_crop_flip_seed2026"
    },
    "D": {
        42: "cnn_crop_flip_cutout_v1",
        123: "cnn_crop_flip_cutout_seed123",
        2026: "cnn_crop_flip_cutout_seed2026"
    }
}

results = {"B": {}, "D": {}}

for method, seed_runs in runs.items():
    for seed, name in seed_runs.items():
        folder = Path("experiments") / name

        with (folder / "history.csv").open() as f:
            history = list(csv.DictReader(f))

        best = max(
            history,
            key=lambda r: float(r["val_accuracy"])
        )

        summary = json.loads(
            (folder / "summary.json").read_text()
        )

        results[method][seed] = {
            "best_accuracy": float(best["val_accuracy"]),
            "best_epoch": int(best["epoch"]),
            "training_minutes": summary["training_seconds"] / 60
        }

aggregate = {}

for method in ["B", "D"]:
    values = [
        results[method][seed]["best_accuracy"]
        for seed in [42, 123, 2026]
    ]

    times = [
        results[method][seed]["training_minutes"]
        for seed in [42, 123, 2026]
    ]

    aggregate[method] = {
        "mean_accuracy": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "mean_training_minutes": statistics.mean(times),
        "individual_results": results[method]
    }

differences = {
    seed: (
        results["D"][seed]["best_accuracy"]
        - results["B"][seed]["best_accuracy"]
    )
    for seed in [42, 123, 2026]
}

aggregate["paired_difference"] = {
    "mean_pp": statistics.mean(differences.values()),
    "sample_std_pp": statistics.stdev(differences.values()),
    "per_seed": differences
}

Path("experiments/multiseed_summary.json").write_text(
    json.dumps(aggregate, indent=4)
)

with Path("experiments/multiseed_summary.csv").open(
    "w", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "method", "seed", "best_accuracy",
        "best_epoch", "training_minutes"
    ])

    for method in ["B", "D"]:
        for seed in [42, 123, 2026]:
            r = results[method][seed]
            writer.writerow([
                method, seed, r["best_accuracy"],
                r["best_epoch"], r["training_minutes"]
            ])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for seed in [42, 123, 2026]:
    axes[0].plot(
        ["B", "D"],
        [
            results["B"][seed]["best_accuracy"],
            results["D"][seed]["best_accuracy"]
        ],
        marker="o",
        label=f"Seed {seed}"
    )

axes[0].set_title("Paired Results Across Random Seeds")
axes[0].set_ylabel("Best Validation Accuracy (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

means = [
    aggregate[m]["mean_accuracy"] for m in ["B", "D"]
]
stds = [
    aggregate[m]["sample_std"] for m in ["B", "D"]
]

axes[1].errorbar(
    ["B: Crop + Flip", "D: + Cutout"],
    means,
    yerr=stds,
    fmt="o",
    capsize=6
)

for x, method in enumerate(["B", "D"]):
    values = [
        results[method][seed]["best_accuracy"]
        for seed in [42, 123, 2026]
    ]
    axes[1].scatter([x] * 3, values)

axes[1].set_title("Mean Accuracy and Sample Standard Deviation")
axes[1].set_ylabel("Best Validation Accuracy (%)")
axes[1].grid(alpha=0.3)

fig.tight_layout()

output = Path("outputs/figures/multiseed_comparison.png")
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("===== MULTI-SEED ANALYSIS =====")

for method in ["B", "D"]:
    r = aggregate[method]
    print(
        f"{method}: "
        f"{r['mean_accuracy']:.3f}% "
        f"+/- {r['sample_std']:.3f} pp"
    )
    print(
        "Mean training time:",
        f"{r['mean_training_minutes']:.2f} min"
    )

print("\nPaired D-B differences:", differences)
print("Mean paired difference:",
      f"{aggregate['paired_difference']['mean_pp']:.3f} pp")

print("\nSaved:", output)
print("Saved: experiments/multiseed_summary.csv")
print("Saved: experiments/multiseed_summary.json")
