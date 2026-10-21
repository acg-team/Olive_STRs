#!/usr/bin/env python3
"""Fit matched-indel slopes and compare them with current-setting STR slopes."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT.parent / "03_embeddings"
HERE = ROOT / "matched_indel_control"
INPUT = ROOT / "shared_inputs"
MODELS = {
    "ArgoNT": {
        "key": "argont", "layers": [11, 21, 31, 40], "primary": 21, "color": "#0072B2",
        "regressions": EMBEDDINGS / "agront/results/per_locus_regressions.csv.gz",
    },
    "PlantCAD2": {
        "key": "plantcad2", "layers": [12, 24, 36, 48], "primary": 36, "color": "#E69F00",
        "regressions": EMBEDDINGS / "plantcad2/results/per_locus_regressions.csv.gz",
    },
    "GPN": {
        "key": "gpn", "layers": [25], "primary": 25, "color": "#009E73",
        "regressions": EMBEDDINGS / "gpn/results/per_locus_regressions.csv.gz",
    },
}
POOLINGS = ("whole_mean", "str_mean")
METRICS = ("l1", "l2", "cosine")
SEED = 20260724


def set_style():
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 17,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def regression(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    ssx = xc @ xc
    if ssx <= 0:
        raise RuntimeError("Predictor has zero variance")
    slope = (xc @ yc) / ssx
    intercept = y.mean() - slope * x.mean()
    fitted = x * slope + intercept
    total = yc @ yc
    r2 = np.nan if total <= 0 else 1 - np.sum((y - fitted) ** 2) / total
    return slope, intercept, r2


def distance(matrix, reference, alternates, metric):
    ref = matrix[reference].astype(np.float64)
    alt = matrix[alternates].astype(np.float64)
    delta = alt - ref
    if metric == "l1":
        return np.abs(delta).sum(axis=1)
    if metric == "l2":
        return np.linalg.norm(delta, axis=1)
    denominator = np.maximum(np.linalg.norm(alt, axis=1) * np.linalg.norm(ref), 1e-12)
    return np.maximum(1 - alt @ ref / denominator, 0)


def load_replicate(model, replicate):
    pieces = []
    for shard in range(10):
        path = HERE / "results/embeddings" / f"{model}_control_r{replicate}_shard_{shard:04d}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path) as handle:
            whole = handle["whole_mean"][:]
            region = handle["str_mean"][:]
            if whole.ndim == 2:
                whole = whole[:, None, :]
                region = region[:, None, :]
            pieces.append({
                "allele_id": handle["allele_id"].asstr()[:],
                "str_id": handle["str_id"].asstr()[:],
                "repeats": handle["allele_repeats"][:],
                "observed": handle["observed"][:].astype(bool),
                "reference": handle["is_reference"][:].astype(bool),
                "whole_mean": whole,
                "str_mean": region,
                "layers": list(map(int, handle.attrs["layers"])),
            })
    layers = pieces[0]["layers"]
    if any(piece["layers"] != layers for piece in pieces):
        raise RuntimeError(f"{model}: layers differ among shards")
    return {
        key: np.concatenate([piece[key] for piece in pieces])
        for key in ("allele_id", "str_id", "repeats", "observed", "reference", "whole_mean", "str_mean")
    } | {"layers": layers}


def fit_controls(model_name, replicate, periods):
    settings = MODELS[model_name]
    data = load_replicate(settings["key"], replicate)
    if data["layers"] != settings["layers"]:
        raise RuntimeError(f"{model_name}: unexpected layers")
    rows = []
    for locus in dict.fromkeys(data["str_id"]):
        indexes = np.flatnonzero(data["str_id"] == locus)
        reference = np.flatnonzero(data["reference"][indexes])
        if len(reference) != 1:
            raise RuntimeError(f"{locus}: reference count={len(reference)}")
        reference = int(reference[0])
        alternates = np.flatnonzero(data["observed"][indexes] & ~data["reference"][indexes])
        alternate_repeats = data["repeats"][indexes][alternates]
        if len(np.unique(alternate_repeats)) < 3:
            continue
        x_repeat = np.abs(
            alternate_repeats - data["repeats"][indexes][reference]
        ).astype(float)
        for pooling in POOLINGS:
            vectors = data[pooling][indexes]
            for layer_column, layer in enumerate(data["layers"]):
                matrix = vectors[:, layer_column]
                for metric in METRICS:
                    y = distance(matrix, reference, alternates, metric)
                    slope, intercept, r2 = regression(x_repeat, y)
                    for predictor, scale in (
                        ("absolute_repeat_difference_from_reference", 1),
                        ("absolute_bp_difference_from_reference", periods[locus]),
                    ):
                        rows.append({
                            "model": model_name, "replicate": replicate, "str_id": locus,
                            "pooling": pooling, "layer": layer, "metric": metric,
                            "predictor": predictor, "n_alternate_alleles": len(alternates),
                            "slope": slope / scale, "intercept": intercept, "r2": r2,
                        })
    return rows


def bootstrap_ci(values, rng, n=10000):
    values = np.asarray(values, float)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def sign_flip(values, rng, n=100000):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    exceed = 0
    chunk = 5000
    for start in range(0, n, chunk):
        count = min(chunk, n - start)
        null = (rng.choice((-1, 1), size=(count, len(values))) * values).mean(axis=1)
        exceed += np.sum(np.abs(null) >= observed)
    return (exceed + 1) / (n + 1)


def main():
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    (HERE / "report").mkdir(parents=True, exist_ok=True)
    selected = set(pd.read_csv(INPUT / "selected_loci.csv").str_id)
    periods = pd.read_csv(INPUT / "loci.csv.gz", usecols=["str_id", "period"]).set_index("str_id").period.to_dict()

    control_rows = []
    for model in MODELS:
        for replicate in (1, 2, 3):
            control_rows.extend(fit_controls(model, replicate, periods))
            print(f"fitted={model} replicate={replicate}", flush=True)
    controls = pd.DataFrame(control_rows)
    controls.to_csv(HERE / "results/control_per_locus_regressions.csv.gz",
                    index=False, compression="gzip")
    means = controls.groupby(
        ["model", "str_id", "pooling", "layer", "metric", "predictor"], as_index=False
    ).agg(control_mean_slope=("slope", "mean"), control_sd_slope=("slope", "std"),
          control_mean_r2=("r2", "mean"))

    baseline_frames = []
    columns = ["str_id", "pooling", "layer", "metric", "predictor", "slope", "r2"]
    for model, settings in MODELS.items():
        frame = pd.read_csv(settings["regressions"], usecols=columns)
        frame = frame[frame.str_id.isin(selected)].copy()
        frame["model"] = model
        bp = frame.copy()
        bp["predictor"] = "absolute_bp_difference_from_reference"
        bp["slope"] = bp.slope / bp.str_id.map(periods)
        frame["predictor"] = "absolute_repeat_difference_from_reference"
        baseline_frames += [frame, bp]
    baseline = pd.concat(baseline_frames, ignore_index=True).rename(
        columns={"slope": "str_slope", "r2": "str_r2"}
    )
    compare = baseline.merge(
        means,
        on=["model", "str_id", "pooling", "layer", "metric", "predictor"],
        validate="one_to_one",
    )
    compare["slope_difference"] = compare.str_slope - compare.control_mean_slope
    compare["slope_ratio"] = compare.str_slope / compare.control_mean_slope.replace(0, np.nan)
    compare.to_csv(HERE / "results/str_vs_control_per_locus.csv.gz",
                   index=False, compression="gzip")

    rng = np.random.default_rng(SEED)
    summaries = []
    group_keys = ["model", "pooling", "layer", "metric", "predictor"]
    for key, frame in compare.groupby(group_keys):
        model, pooling, layer, metric, predictor = key
        is_primary = (
            pooling == "whole_mean"
            and metric == "l2"
            and predictor in (
                "absolute_repeat_difference_from_reference",
                "absolute_bp_difference_from_reference",
            )
            and layer == MODELS[model]["primary"]
        )
        ci = (
            bootstrap_ci(frame.slope_difference, rng)
            if is_primary else (np.nan, np.nan)
        )
        summaries.append(dict(zip(group_keys, key)) | {
            "n_loci": len(frame), "mean_str_slope": frame.str_slope.mean(),
            "mean_control_slope": frame.control_mean_slope.mean(),
            "mean_slope_difference": frame.slope_difference.mean(),
            "median_slope_difference": frame.slope_difference.median(),
            "difference_ci_low": ci[0], "difference_ci_high": ci[1],
            "fraction_str_greater": np.mean(frame.slope_difference > 0),
            "pearson_correlation": frame[["str_slope", "control_mean_slope"]].corr().iloc[0, 1],
            "sign_flip_p_value": (
                sign_flip(frame.slope_difference.to_numpy(), rng)
                if is_primary else np.nan
            ),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "results/str_vs_control_summary.csv", index=False)

    set_style()
    primary_frames = {}
    predictors = {
        "per_repeat": {
            "predictor": "absolute_repeat_difference_from_reference",
            "unit": "repeat",
            "suffix": "",
        },
        "per_bp": {
            "predictor": "absolute_bp_difference_from_reference",
            "unit": "bp",
            "suffix": "_per_bp",
        },
    }
    for scale, options in predictors.items():
        primary = compare[
            (compare.pooling == "whole_mean") & (compare.metric == "l2") &
            (compare.predictor == options["predictor"]) &
            compare.apply(lambda x: x.layer == MODELS[x.model]["primary"], axis=1)
        ].copy()
        if primary.groupby("model").str_id.nunique().to_dict() != {x: 1000 for x in MODELS}:
            raise RuntimeError(f"{scale} comparison does not contain 1,000 loci per model")
        primary_frames[scale] = primary

        for model, frame in primary.groupby("model"):
            color = MODELS[model]["color"]
            limits = [
                min(frame.control_mean_slope.min(), frame.str_slope.min()),
                max(frame.control_mean_slope.max(), frame.str_slope.max()),
            ]
            padding = 0.04 * max(limits[1] - limits[0], 1e-12)
            limits = [limits[0] - padding, limits[1] + padding]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(frame.control_mean_slope, frame.str_slope, color=color, alpha=0.65,
                       s=38, edgecolor="none")
            ax.plot(limits, limits, "--", color="black", linewidth=1.5)
            ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"Matched-indel mean L2 slope per {options['unit']}")
            ax.set_ylabel(f"{model} STR L2 slope per {options['unit']}")
            ax.set_title(f"{model}: STR versus matched indel")
            fig.tight_layout()
            stem = (
                f"{MODELS[model]['key']}_str_vs_matched_indel_l2_slopes"
                f"{options['suffix']}"
            )
            fig.savefig(HERE / f"figures/{stem}.png", dpi=300)
            fig.savefig(HERE / f"figures/{stem}.pdf")
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
        for ax, model in zip(axes, MODELS):
            frame = primary[primary.model == model]
            limits = [
                min(frame.control_mean_slope.min(), frame.str_slope.min()),
                max(frame.control_mean_slope.max(), frame.str_slope.max()),
            ]
            padding = 0.04 * max(limits[1] - limits[0], 1e-12)
            limits = [limits[0] - padding, limits[1] + padding]
            ax.scatter(frame.control_mean_slope, frame.str_slope, color=MODELS[model]["color"],
                       alpha=0.6, s=28, edgecolor="none")
            ax.plot(limits, limits, "--", color="black", linewidth=1.3)
            ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
            ax.set_title(model)
            ax.set_xlabel(f"Matched-indel mean L2 slope per {options['unit']}")
            ax.set_ylabel(f"STR L2 slope per {options['unit']}")
        fig.tight_layout()
        stem = f"three_model_str_vs_matched_indel_l2_slopes{options['suffix']}"
        fig.savefig(HERE / f"figures/{stem}.png", dpi=300)
        fig.savefig(HERE / f"figures/{stem}.pdf")
        plt.close(fig)

    per_bp = primary_frames["per_bp"]
    category_colors = {"STR": "#0072B2", "Matched-indel": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, model in zip(axes, MODELS):
        frame = per_bp[per_bp.model == model]
        for category_index, (category, column) in enumerate((
            ("STR", "str_slope"),
            ("Matched-indel", "control_mean_slope"),
        )):
            violin = ax.violinplot(
                frame[column].dropna().to_numpy(),
                positions=[category_index],
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
                points=150,
            )
            for body in violin["bodies"]:
                body.set_facecolor(category_colors[category])
                body.set_edgecolor("white")
                body.set_linewidth(0.8)
                body.set_alpha(0.78)
            violin["cmedians"].set_color("black")
            violin["cmedians"].set_linewidth(1.5)
        ax.axhline(0, color="#888888", linewidth=0.9, zorder=0)
        ax.set_xticks([0, 1], ["STR", "Matched-indel"])
        ax.set_xlabel("")
        ax.set_ylabel("Embedding sensitivity (per bp)")
        ax.set_title(model)
    fig.tight_layout()
    stem = "three_model_str_vs_matched_indel_l2_slope_per_bp_violin"
    fig.savefig(HERE / f"figures/{stem}.png", dpi=300)
    fig.savefig(HERE / f"figures/{stem}.pdf")
    plt.close(fig)

    main_summary = summary[
        (summary.pooling == "whole_mean") & (summary.metric == "l2") &
        summary.predictor.isin([x["predictor"] for x in predictors.values()]) &
        summary.apply(lambda x: x.layer == MODELS[x.model]["primary"], axis=1)
    ]
    lines = [
        "# Current-setting matched-indel robustness control", "",
        "Genuine STR slopes from the current genome-wide sequence construction are compared "
        "with the mean slope from three deterministic, nested, length-matched nonrepetitive "
        "indel series at the same loci.", "",
        "## Matched-indel construction", "",
        "The matched indel is an in-silico nonrepetitive sequence series rather than "
        "another STR. For each STR locus and each of three control replicates, A/C/G/T "
        "bases from the left and right non-STR flanks were pooled and deterministically "
        "shuffled using a seed derived from the locus ID and replicate. Candidate "
        "sequences containing at least four tandem copies of any 1–6 bp motif were "
        "rejected.", "",
        "One control tape was generated per locus and replicate, with sufficient length "
        "for the longest allele. For an STR allele with \\(r\\) repeats and motif length "
        "\\(p\\), the first \\(r \\times p\\) bases of the tape were used. Control alleles "
        "were therefore nested prefixes: any pairwise length difference was exactly the "
        "same as for the corresponding STR alleles, while the inserted sequence was "
        "nonrepetitive. Alleles shorter than the reference represented matched deletions, "
        "and alleles longer than the reference represented matched insertions.", "",
        "ArgoNT and PlantCAD2 controls retained the current sequence construction: "
        "500-bp left flank, control allele, and 500-bp right flank. GPN controls retained "
        "a 256-bp left flank and adjusted the right flank so every input remained 512 bp. "
        "Within each replicate, embedding distances were anchored to the length-matched "
        "control reference allele. Slopes were fitted per repeat and per bp, then averaged "
        "across the three control replicates before comparison with the current-setting "
        "STR slopes.", "",
    ]
    for scale, options in predictors.items():
        lines += [f"## L2 slope per {options['unit']}", ""]
        current = main_summary[main_summary.predictor == options["predictor"]]
        for row in current.itertuples(index=False):
            lines += [
                f"### {row.model}, layer {row.layer}", "",
                f"- Loci: {row.n_loci:,}",
                f"- Mean STR slope: {row.mean_str_slope:.6g}",
                f"- Mean matched-indel slope: {row.mean_control_slope:.6g}",
                f"- Mean paired difference: {row.mean_slope_difference:.6g}",
                f"- 95% locus-bootstrap CI: {row.difference_ci_low:.6g} to {row.difference_ci_high:.6g}",
                f"- STR slope greater than control: {row.fraction_str_greater:.1%}",
                f"- Paired sign-flip p-value: {row.sign_flip_p_value:.4g}",
                f"- Slope correlation: {row.pearson_correlation:.3f}", "",
            ]
    (HERE / "report/REPORT.md").write_text("\n".join(lines) + "\n")
    payload = {
        "status": "PASS", "models": 3, "replicates": 3, "loci_per_model": 1000,
        "primary_rows_per_scale": {
            scale: len(frame) for scale, frame in primary_frames.items()
        },
    }
    (HERE / "results/validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
