#!/usr/bin/env python3
"""Compare current and fixed-total whole-mean embeddings directly."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT.parent / "03_embeddings"
HERE = ROOT / "fixed_total_embedding_comparison"
INPUT = ROOT / "shared_inputs"
MODELS = {
    "ArgoNT": {
        "key": "argont", "layers": [11, 21, 31, 40], "primary": 21,
        "baseline": EMBEDDINGS / "agront/results/embeddings/master.h5",
        "color": "#0072B2",
    },
    "PlantCAD2": {
        "key": "plantcad2", "layers": [12, 24, 36, 48], "primary": 36,
        "baseline": EMBEDDINGS / "plantcad2/results/embeddings/master.h5",
        "color": "#E69F00",
    },
}


def set_style():
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def ecdf(ax, values, label, color):
    x = np.sort(np.asarray(values))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", label=label, color=color, linewidth=2.2)


def load_alternative(key: str):
    frames = []
    for shard in range(10):
        path = HERE / "results/embeddings" / f"{key}_fixed_total_shard_{shard:04d}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path) as handle:
            frames.append((
                handle["allele_id"].asstr()[:],
                handle["whole_mean"][:],
                list(map(int, handle.attrs["layers"])),
            ))
    layers = frames[0][2]
    if any(item[2] != layers for item in frames):
        raise RuntimeError(f"{key}: layer mismatch among shards")
    return np.concatenate([x[0] for x in frames]), np.concatenate([x[1] for x in frames]), layers


def take_rows(dataset, indexes):
    indexes = np.asarray(indexes)
    order = np.argsort(indexes)
    sorted_values = dataset[indexes[order]]
    reverse = np.empty_like(order)
    reverse[order] = np.arange(len(order))
    return sorted_values[reverse]


def compare_vectors(current, alternative):
    delta = alternative.astype(np.float64) - current.astype(np.float64)
    current64 = current.astype(np.float64)
    alternative64 = alternative.astype(np.float64)
    norm_current = np.linalg.norm(current64, axis=1)
    norm_alternative = np.linalg.norm(alternative64, axis=1)
    cosine = np.sum(current64 * alternative64, axis=1) / np.maximum(
        norm_current * norm_alternative, 1e-12
    )
    relative_l2 = np.linalg.norm(delta, axis=1) / np.maximum(norm_current, 1e-12)
    mean_absolute = np.abs(delta).mean(axis=1)
    current_centered = current64 - current64.mean(axis=1, keepdims=True)
    alternative_centered = alternative64 - alternative64.mean(axis=1, keepdims=True)
    pearson = np.sum(current_centered * alternative_centered, axis=1) / np.maximum(
        np.linalg.norm(current_centered, axis=1) * np.linalg.norm(alternative_centered, axis=1),
        1e-12,
    )
    return cosine, relative_l2, mean_absolute, pearson


def main():
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    (HERE / "report").mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(INPUT / "alleles.csv.gz")
    loci = pd.read_csv(INPUT / "loci.csv.gz", usecols=["str_id", "period"])
    metadata = metadata.merge(loci, on="str_id", validate="many_to_one")
    reference_repeats = metadata.loc[metadata.is_reference.astype(bool), ["str_id", "allele_repeats"]].rename(
        columns={"allele_repeats": "reference_repeats"}
    )
    metadata = metadata.merge(reference_repeats, on="str_id", validate="many_to_one")
    metadata["absolute_repeat_difference"] = (
        metadata.allele_repeats - metadata.reference_repeats
    ).abs()
    metadata["absolute_bp_difference"] = metadata.absolute_repeat_difference * metadata.period
    meta = metadata.set_index("allele_id")

    rows = []
    for model, settings in MODELS.items():
        allele_ids, alternative, layers = load_alternative(settings["key"])
        if layers != settings["layers"]:
            raise RuntimeError(f"{model}: layers {layers} != {settings['layers']}")
        if len(set(allele_ids)) != len(allele_ids):
            raise RuntimeError(f"{model}: duplicate alternative allele IDs")
        with h5py.File(settings["baseline"]) as baseline:
            baseline_ids = baseline["allele_id"].asstr()[:]
            index = {allele_id: i for i, allele_id in enumerate(baseline_ids)}
            try:
                baseline_indexes = [index[x] for x in allele_ids]
            except KeyError as error:
                raise RuntimeError(f"{model}: baseline lacks {error.args[0]}") from error
            current = take_rows(baseline["whole_mean"], baseline_indexes)
            baseline_layers = list(map(int, baseline.attrs["layers"]))
        if baseline_layers != layers or current.shape != alternative.shape:
            raise RuntimeError(f"{model}: baseline/alternative shape or layer mismatch")
        for column, layer in enumerate(layers):
            cosine, relative_l2, mean_absolute, pearson = compare_vectors(
                current[:, column], alternative[:, column]
            )
            for i, allele_id in enumerate(allele_ids):
                item = meta.loc[allele_id]
                rows.append({
                    "model": model, "layer": layer, "allele_id": allele_id,
                    "str_id": item.str_id, "allele_repeats": int(item.allele_repeats),
                    "reference_repeats": int(item.reference_repeats),
                    "is_reference": bool(item.is_reference),
                    "observed": bool(item.observed), "motif_length": int(item.period),
                    "absolute_repeat_difference": int(item.absolute_repeat_difference),
                    "absolute_bp_difference": int(item.absolute_bp_difference),
                    "cosine_similarity": cosine[i], "relative_l2": relative_l2[i],
                    "mean_absolute_difference": mean_absolute[i],
                    "pearson_correlation": pearson[i],
                })
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(HERE / "results/current_vs_fixed_total_per_allele.csv.gz",
                       index=False, compression="gzip")

    metrics = ["cosine_similarity", "relative_l2", "mean_absolute_difference", "pearson_correlation"]
    summary_rows = []
    for (model, layer), frame in comparisons.groupby(["model", "layer"]):
        row = {"model": model, "layer": layer, "n_alleles": len(frame)}
        for metric in metrics:
            values = frame[metric].to_numpy()
            row |= {
                f"{metric}_median": np.median(values),
                f"{metric}_q1": np.quantile(values, 0.25),
                f"{metric}_q3": np.quantile(values, 0.75),
                f"{metric}_p95": np.quantile(values, 0.95),
            }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(HERE / "results/embedding_agreement_summary.csv", index=False)

    stratified = comparisons[
        comparisons.apply(lambda x: x.layer == MODELS[x.model]["primary"], axis=1)
    ].groupby(["model", "motif_length", "absolute_bp_difference"])[metrics].agg(
        ["count", "median", "mean"]
    )
    stratified.to_csv(HERE / "results/primary_layer_stratified_summary.csv")

    set_style()
    display = comparisons[comparisons.apply(
        lambda x: x.layer == MODELS[x.model]["primary"], axis=1
    )].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    labels = {
        "cosine_similarity": "Cosine similarity",
        "relative_l2": "Relative L2 difference",
        "mean_absolute_difference": "Mean absolute difference",
        "pearson_correlation": "Pearson correlation",
    }
    palette = {model: settings["color"] for model, settings in MODELS.items()}
    for ax, metric in zip(axes.flat, metrics):
        for model, frame in display.groupby("model"):
            ecdf(ax, frame[metric], model, palette[model])
        ax.set_xlabel(labels[metric]); ax.set_ylabel("Fraction of alleles")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures/primary_layer_embedding_agreement_distributions.png", dpi=300)
    fig.savefig(HERE / "figures/primary_layer_embedding_agreement_distributions.pdf")
    plt.close(fig)

    long = summary.melt(
        id_vars=["model", "layer"], value_vars=[f"{x}_median" for x in metrics],
        var_name="metric", value_name="median",
    )
    long["metric"] = long.metric.str.removesuffix("_median").map(labels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, metric in zip(axes.flat, labels.values()):
        frame = long[long.metric == metric]
        for model, group in frame.groupby("model"):
            ax.plot(group.layer, group["median"], marker="o", linewidth=2.2,
                    color=palette[model], label=model)
        ax.set_title(metric); ax.set_xlabel("Hidden layer"); ax.set_ylabel("Median")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures/embedding_agreement_by_layer.png", dpi=300)
    fig.savefig(HERE / "figures/embedding_agreement_by_layer.pdf")
    plt.close(fig)

    primary = summary[summary.apply(lambda x: x.layer == MODELS[x.model]["primary"], axis=1)]
    lines = [
        "# Fixed-total context embedding robustness",
        "",
        "This test directly compares each allele's current whole-mean embedding with its "
        "fixed-500-bp-left, fixed-1,000-bp-total embedding. No regression or significance "
        "testing is performed.",
        "",
        "## Primary layers",
        "",
    ]
    for row in primary.itertuples(index=False):
        lines += [
            f"### {row.model}, layer {row.layer}",
            "",
            f"- Alleles: {row.n_alleles:,}",
            f"- Median cosine similarity: {row.cosine_similarity_median:.6g}",
            f"- Median relative L2 difference: {row.relative_l2_median:.6g}",
            f"- Median mean absolute difference: {row.mean_absolute_difference_median:.6g}",
            f"- Median Pearson correlation: {row.pearson_correlation_median:.6g}",
            "",
        ]
    (HERE / "report/REPORT.md").write_text("\n".join(lines) + "\n")
    payload = {"status": "PASS", "models": 2, "loci": 1000, "alleles": int(metadata.shape[0])}
    (HERE / "results/validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
