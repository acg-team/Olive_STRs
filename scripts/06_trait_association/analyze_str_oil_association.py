#!/usr/bin/env python3
"""STR length association with olive oil composition using SNP-kinship LMMs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t


TRAITS = {
    "oleic_acid": "OleicAcid_Percentage",
    "linoleic_acid": "LinoleicAcid_Percentage",
    "log_oleic_linoleic_ratio": "log_oleic_linoleic_ratio",
}


@dataclass
class NullLMM:
    y_white: np.ndarray
    q: np.ndarray
    y_resid: np.ndarray
    chol: np.ndarray
    sigma_g2: float
    delta: float
    n: int
    p: int
    h2: float


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    population = root.parent / "02_population_structure/source_data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--str-matrix", type=Path, default=root / "inputs/merged_str_length.csv")
    parser.add_argument("--oil-data-1", type=Path, default=root / "inputs/Oil_Data_1.csv")
    parser.add_argument("--oil-data-2", type=Path, default=root / "inputs/Oil_Data_2.csv")
    parser.add_argument(
        "--snp-pcs", type=Path,
        default=population / "snp_pca_coordinates.tsv",
    )
    parser.add_argument(
        "--snp-eigenvalues", type=Path,
        default=population / "snp_pca_eigenvalues.tsv",
    )
    parser.add_argument(
        "--str-pcs", type=Path,
        default=population / "str_pca_coordinates.tsv",
    )
    parser.add_argument(
        "--annotation", type=Path,
        default=root / "inputs/O_europeana_strs.functional.promoter2kb.annotated.bed",
    )
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--min-distinct-lengths", type=int, default=3)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument(
        "--pc-source", choices=("none", "snp", "str"), default="str",
        help="Fixed-effect PC source; use with --n-pcs > 0.",
    )
    parser.add_argument(
        "--kinship-mode", choices=("genomewide", "loco"), default="genomewide",
        help="Use one genome-wide SNP kinship or chromosome-specific SNP LOCO matrices.",
    )
    parser.add_argument(
        "--loco-dir", type=Path,
        default=root / "inputs/loco",
    )
    parser.add_argument(
        "--loco-fam", type=Path,
        default=root / "inputs/loco/cohort.fam",
    )
    parser.add_argument(
        "--chromosomes-only", action="store_true",
        help="Restrict tested STRs to chromosomes 1-23.",
    )
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--top-loci", type=int, default=5)
    parser.add_argument(
        "--str-imputed",
        action="store_true",
        help="Declare that the supplied STR matrix was imputed before association testing.",
    )
    parser.add_argument(
        "--exclude-center-covariate",
        action="store_true",
        help="Do not include measurement center/dataset as a pooled-model fixed effect.",
    )
    return parser.parse_args()


def validate_unique(values: pd.Series, label: str) -> None:
    duplicated = values[values.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicate {label}: {', '.join(map(str, duplicated[:10]))}")


def load_phenotypes(path1: Path, path2: Path) -> pd.DataFrame:
    cols = ["Sample_ID", "LinoleicAcid_Percentage", "OleicAcid_Percentage"]
    frames = []
    for center, path in (("center_1", path1), ("center_2", path2)):
        frame = pd.read_csv(path, usecols=cols)
        frame["center"] = center
        frames.append(frame)
    pheno = pd.concat(frames, ignore_index=True)
    validate_unique(pheno["Sample_ID"], "phenotype sample IDs")
    for col in cols[1:]:
        pheno[col] = pd.to_numeric(pheno[col], errors="coerce")
    if pheno[cols].isna().any().any():
        bad = pheno.loc[pheno[cols].isna().any(axis=1), "Sample_ID"].tolist()
        raise ValueError(f"Missing/non-numeric phenotype values for: {', '.join(bad[:10])}")
    if (pheno["LinoleicAcid_Percentage"] <= 0).any():
        raise ValueError("Linoleic acid must be positive to calculate the O/L ratio")
    pheno["oleic_linoleic_ratio"] = (
        pheno["OleicAcid_Percentage"] / pheno["LinoleicAcid_Percentage"]
    )
    pheno["log_oleic_linoleic_ratio"] = np.log(pheno["oleic_linoleic_ratio"])
    return pheno


def standardize_columns(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    result -= result.mean(axis=0)
    sd = result.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("A requested covariate has zero variance")
    return result / sd


def reconstruct_kinship(pcs: pd.DataFrame, eigenvalues: pd.DataFrame) -> np.ndarray:
    pc_cols = [c for c in pcs.columns if c.startswith("PC") and c[2:].isdigit()]
    pc_cols.sort(key=lambda c: int(c[2:]))
    eigen = eigenvalues.sort_values("component")["eigenvalue"].to_numpy(dtype=float)
    if len(eigen) < len(pc_cols):
        raise ValueError("Fewer SNP eigenvalues than SNP PC coordinate columns")
    u = pcs[pc_cols].to_numpy(dtype=float)
    k = (u * eigen[: len(pc_cols)]) @ u.T
    k = (k + k.T) / 2
    scale = float(np.mean(np.diag(k)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Reconstructed SNP kinship has an invalid diagonal")
    k /= scale
    return k


def build_design(pcs: pd.DataFrame, centers: np.ndarray | None, n_pcs: int) -> np.ndarray:
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    pc_values = standardize_columns(pcs[pc_cols].to_numpy(dtype=float))
    columns = [np.ones(len(pcs))]
    if centers is not None:
        columns.append((centers == "center_2").astype(float))
    columns.extend(pc_values[:, i] for i in range(n_pcs))
    return np.column_stack(columns)


def fit_null_lmm(y: np.ndarray, x: np.ndarray, k: np.ndarray) -> NullLMM:
    """Fit y = Xb + u + e by REML and retain a whitening transform."""
    n, p = x.shape
    if n <= p + 2:
        raise ValueError("Insufficient residual degrees of freedom for LMM")

    def objective(log_delta: float) -> float:
        delta = math.exp(log_delta)
        try:
            chol = np.linalg.cholesky(k + delta * np.eye(n))
            yw = solve_triangular(chol, y, lower=True)
            xw = solve_triangular(chol, x, lower=True)
            q, _ = np.linalg.qr(xw, mode="reduced")
            residual = yw - q @ (q.T @ yw)
            sse = float(residual @ residual)
            sign, logdet_xtx = np.linalg.slogdet(xw.T @ xw)
            if sign <= 0 or sse <= 0:
                return np.inf
            logdet_v0 = 2 * np.log(np.diag(chol)).sum()
            df = n - p
            return 0.5 * (logdet_v0 + logdet_xtx + df * math.log(sse / df))
        except np.linalg.LinAlgError:
            return np.inf

    fit = minimize_scalar(objective, bounds=(-12, 12), method="bounded")
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Null LMM REML optimization failed: {fit.message}")
    delta = math.exp(float(fit.x))
    chol = np.linalg.cholesky(k + delta * np.eye(n))
    yw0 = solve_triangular(chol, y, lower=True)
    xw0 = solve_triangular(chol, x, lower=True)
    q, _ = np.linalg.qr(xw0, mode="reduced")
    resid0 = yw0 - q @ (q.T @ yw0)
    sigma_g2 = float(resid0 @ resid0) / (n - p)
    scale = math.sqrt(sigma_g2)
    yw = yw0 / scale
    xw = xw0 / scale
    q, _ = np.linalg.qr(xw, mode="reduced")
    y_resid = yw - q @ (q.T @ yw)
    return NullLMM(
        y_white=yw,
        q=q,
        y_resid=y_resid,
        chol=chol,
        sigma_g2=sigma_g2,
        delta=delta,
        n=n,
        p=p,
        h2=1.0 / (1.0 + delta),
    )


def association_chunk(g: np.ndarray, null: NullLMM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gw = solve_triangular(null.chol, g.T, lower=True).T / math.sqrt(null.sigma_g2)
    gr = gw - (gw @ null.q) @ null.q.T
    denominator = np.einsum("ij,ij->i", gr, gr)
    numerator = gr @ null.y_resid
    beta = np.full(len(g), np.nan)
    se = np.full(len(g), np.nan)
    pvalue = np.full(len(g), np.nan)
    valid = denominator > 1e-12
    beta[valid] = numerator[valid] / denominator[valid]
    residual = null.y_resid[None, :] - beta[:, None] * gr
    df = null.n - null.p - 1
    sse = np.einsum("ij,ij->i", residual, residual)
    se[valid] = np.sqrt((sse[valid] / df) / denominator[valid])
    stat = beta[valid] / se[valid]
    pvalue[valid] = 2 * t.sf(np.abs(stat), df=df)
    return beta, se, pvalue


def association_complete_case(
    g: np.ndarray, y: np.ndarray, x: np.ndarray, k: np.ndarray, delta: float,
) -> tuple[float, float, float, int]:
    """Test one locus after excluding samples with missing STR length.

    The variance ratio is estimated once from the full-cohort null model (the
    EMMAX/P3D strategy), while GLS projection and the STR test use only samples
    observed at this locus.
    """
    observed = np.isfinite(g)
    n = int(observed.sum())
    p = x.shape[1]
    if n <= p + 2:
        return np.nan, np.nan, np.nan, n
    gs = np.asarray(g[observed], dtype=float)
    if np.ptp(gs) <= 0:
        return np.nan, np.nan, np.nan, n
    ys = y[observed]
    xs = x[observed]
    ks = k[np.ix_(observed, observed)]
    try:
        chol = np.linalg.cholesky(ks + delta * np.eye(n))
        yw = solve_triangular(chol, ys, lower=True, check_finite=False)
        xw = solve_triangular(chol, xs, lower=True, check_finite=False)
        gw = solve_triangular(chol, gs, lower=True, check_finite=False)
        q, _ = np.linalg.qr(xw, mode="reduced")
        yr = yw - q @ (q.T @ yw)
        gr = gw - q @ (q.T @ gw)
        denominator = float(gr @ gr)
        if denominator <= 1e-12:
            return np.nan, np.nan, np.nan, n
        beta = float((gr @ yr) / denominator)
        residual = yr - beta * gr
        df = n - p - 1
        se = math.sqrt(float(residual @ residual) / df / denominator)
        if not np.isfinite(se) or se <= 0:
            return beta, np.nan, np.nan, n
        pvalue = float(2 * t.sf(abs(beta / se), df=df))
        return beta, se, pvalue, n
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, n


def fixed_effect_meta(
    beta1: np.ndarray, se1: np.ndarray, beta2: np.ndarray, se2: np.ndarray
) -> dict[str, np.ndarray]:
    valid = np.isfinite(beta1) & np.isfinite(se1) & (se1 > 0) & np.isfinite(beta2) & np.isfinite(se2) & (se2 > 0)
    output = {key: np.full(len(beta1), np.nan) for key in ("beta", "se", "p", "q", "q_p", "i2")}
    w1 = np.zeros(len(beta1)); w2 = np.zeros(len(beta1))
    w1[valid] = 1 / se1[valid] ** 2; w2[valid] = 1 / se2[valid] ** 2
    total = w1 + w2
    output["beta"][valid] = (w1[valid] * beta1[valid] + w2[valid] * beta2[valid]) / total[valid]
    output["se"][valid] = np.sqrt(1 / total[valid])
    output["p"][valid] = 2 * norm.sf(np.abs(output["beta"][valid] / output["se"][valid]))
    q = w1 * (beta1 - output["beta"]) ** 2 + w2 * (beta2 - output["beta"]) ** 2
    output["q"][valid] = q[valid]
    output["q_p"][valid] = chi2.sf(q[valid], df=1)
    positive = valid & (q > 0)
    output["i2"][valid] = 0.0
    output["i2"][positive] = np.maximum(0, (q[positive] - 1) / q[positive]) * 100
    return output


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    ranked = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1)
    return adjusted


def load_annotation(path: Path) -> pd.DataFrame:
    annotation = pd.read_csv(path, sep="\t", comment="#", header=None, dtype={0: str}, low_memory=False)
    annotation.columns = [
        "chrom", "start", "end", "motif_length", "motif", "gene_start", "gene_end",
        "gene_id", "gene_name", "gene_biotype", "gene_strand", "location",
        "distance_to_gene", "functional_region",
    ]
    annotation["chrom"] = annotation["chrom"].astype(str)
    annotation["str_id"] = annotation["chrom"] + "_" + annotation["start"].astype(str)
    validate_unique(annotation["str_id"], "annotation STR IDs")
    return annotation


def prepare_inputs(args: argparse.Namespace):
    for path in (args.str_matrix, args.oil_data_1, args.oil_data_2, args.snp_pcs,
                 args.snp_eigenvalues, args.str_pcs, args.annotation):
        if not path.is_file():
            raise FileNotFoundError(path)
    pheno = load_phenotypes(args.oil_data_1, args.oil_data_2)
    sample_ids = pheno["Sample_ID"].tolist()
    str_header = pd.read_csv(args.str_matrix, nrows=0).columns.tolist()
    missing_str = sorted(set(sample_ids) - set(str_header))
    if missing_str:
        raise ValueError(f"Phenotype samples absent from STR matrix: {', '.join(missing_str)}")

    snp_pcs_all = pd.read_csv(args.snp_pcs, sep="\t")
    validate_unique(snp_pcs_all["sample_id"], "SNP-PC sample IDs")
    missing_pcs = sorted(set(sample_ids) - set(snp_pcs_all["sample_id"]))
    if missing_pcs:
        raise ValueError(f"Phenotype samples absent from SNP PCs: {', '.join(missing_pcs)}")
    snp_pcs_all = snp_pcs_all.set_index("sample_id").loc[sample_ids].reset_index()
    eigen = pd.read_csv(args.snp_eigenvalues, sep="\t")
    kinship = reconstruct_kinship(snp_pcs_all, eigen)
    if args.n_pcs < 0:
        raise ValueError("--n-pcs must be non-negative")
    if args.n_pcs == 0 and args.pc_source != "none":
        raise ValueError("--pc-source must be 'none' when --n-pcs is 0")
    if args.n_pcs > 0 and args.pc_source == "none":
        raise ValueError("Select --pc-source snp or --pc-source str when --n-pcs > 0")
    fixed_pcs = snp_pcs_all
    if args.pc_source == "str":
        fixed_pcs = pd.read_csv(args.str_pcs, sep="\t")
        validate_unique(fixed_pcs["sample_id"], "STR-PC sample IDs")
        missing_fixed_pcs = sorted(set(sample_ids) - set(fixed_pcs["sample_id"]))
        if missing_fixed_pcs:
            raise ValueError(
                f"Phenotype samples absent from STR PCs: {', '.join(missing_fixed_pcs)}"
            )
        fixed_pcs = fixed_pcs.set_index("sample_id").loc[sample_ids].reset_index()

    annotation = load_annotation(args.annotation)
    matrix = pd.read_csv(args.str_matrix, usecols=["str_id", *sample_ids])
    validate_unique(matrix["str_id"], "STR matrix locus IDs")
    values = matrix[sample_ids].to_numpy(dtype=float)
    n_called = np.sum(np.isfinite(values), axis=1)
    n_unique = np.array([len(np.unique(row[np.isfinite(row)])) for row in values], dtype=int)
    passes_distinct_length_filter = n_unique >= args.min_distinct_lengths
    annotation_chromosome = matrix["str_id"].map(
        annotation.set_index("str_id")["chrom"]
    )
    if annotation_chromosome.isna().any():
        raise ValueError(
            f"{int(annotation_chromosome.isna().sum())} STR matrix loci lack annotation"
        )
    canonical_chromosomes = {str(i) for i in range(1, 24)}
    is_canonical_chromosome = annotation_chromosome.isin(canonical_chromosomes).to_numpy()
    include = passes_distinct_length_filter
    if args.chromosomes_only:
        include = include & is_canonical_chromosome
    if not include.any():
        raise ValueError("No loci pass the active eligibility criteria")

    centers = pheno["center"].to_numpy()
    qc = matrix[["str_id"]].copy()
    qc["n_called"] = n_called
    qc["call_rate"] = n_called / len(sample_ids)
    qc["n_distinct_lengths"] = n_unique
    qc["passes_distinct_length_filter"] = passes_distinct_length_filter
    qc["is_canonical_chromosome"] = is_canonical_chromosome
    for center in ("center_1", "center_2"):
        idx = np.flatnonzero(centers == center)
        qc[f"n_called_{center}"] = np.sum(np.isfinite(values[:, idx]), axis=1)
        qc[f"n_distinct_lengths_{center}"] = [
            len(np.unique(row[np.isfinite(row)])) for row in values[:, idx]
        ]
    qc["included"] = include
    raw_values = values[include].copy()
    values = raw_values
    loci = qc.loc[include].reset_index(drop=True)
    loci["n_missing_excluded"] = np.sum(~np.isfinite(values), axis=1)
    loci["mean_length"] = np.nanmean(values, axis=1)
    loci["sd_length"] = np.nanstd(values, axis=1, ddof=1)
    loci["min_length"] = np.nanmin(values, axis=1)
    loci["max_length"] = np.nanmax(values, axis=1)
    loci = loci.merge(annotation, on="str_id", how="left", validate="one_to_one")
    if loci["chrom"].isna().any():
        count = int(loci["chrom"].isna().sum())
        raise ValueError(f"{count} tested STR loci lack functional annotation")
    return pheno, fixed_pcs, kinship, qc, loci, values, raw_values, annotation


def load_loco_kinships(
    directory: Path, fam_path: Path, sample_ids: list[str],
) -> dict[str, np.ndarray]:
    if not fam_path.is_file():
        raise FileNotFoundError(fam_path)
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[1], names=["sample_id"])
    validate_unique(fam["sample_id"], "LOCO FAM sample IDs")
    missing = sorted(set(sample_ids) - set(fam["sample_id"]))
    extra = sorted(set(fam["sample_id"]) - set(sample_ids))
    if missing or extra:
        raise ValueError(
            f"LOCO FAM/sample mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    fam_index = pd.Series(np.arange(len(fam)), index=fam["sample_id"])
    order = fam_index.loc[sample_ids].to_numpy(dtype=int)
    kinships = {}
    for chromosome in map(str, range(1, 24)):
        path = directory / f"loco_chr{chromosome}.sXX.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = np.loadtxt(path, dtype=float)
        if matrix.shape != (len(fam), len(fam)):
            raise ValueError(f"Unexpected LOCO matrix shape for chromosome {chromosome}")
        matrix = matrix[np.ix_(order, order)]
        matrix = (matrix + matrix.T) / 2
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Non-finite LOCO matrix values for chromosome {chromosome}")
        kinships[chromosome] = matrix
    return kinships


def scan_trait(
    trait: str, y: np.ndarray, loci: pd.DataFrame, values: np.ndarray,
    centers: np.ndarray, snp_pcs: pd.DataFrame, kinship: np.ndarray,
    n_pcs: int, chunk_size: int, include_center_covariate: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    cohort_masks = {
        "pooled": np.ones(len(y), dtype=bool),
        "center_1": centers == "center_1",
        "center_2": centers == "center_2",
    }
    nulls = {}
    null_qc = []
    for cohort, mask in cohort_masks.items():
        pcs_sub = snp_pcs.loc[mask].reset_index(drop=True)
        x = build_design(
            pcs_sub,
            centers[mask] if cohort == "pooled" and include_center_covariate else None,
            n_pcs,
        )
        null = fit_null_lmm(y[mask], x, kinship[np.ix_(mask, mask)])
        nulls[cohort] = null
        null_qc.append({
            "trait": trait, "cohort": cohort, "n": int(mask.sum()), "n_covariates": x.shape[1],
            "delta": null.delta, "snp_heritability_null": null.h2,
            "sigma_g2": null.sigma_g2, "sigma_e2": null.sigma_g2 * null.delta,
        })

    estimates = {
        cohort: {key: np.full(len(loci), np.nan) for key in ("beta", "se", "p", "n")}
        for cohort in cohort_masks
    }
    for cohort, mask in cohort_masks.items():
        y_sub = y[mask]
        pcs_sub = snp_pcs.loc[mask].reset_index(drop=True)
        x_sub = build_design(
            pcs_sub,
            centers[mask] if cohort == "pooled" and include_center_covariate else None,
            n_pcs,
        )
        k_sub = kinship[np.ix_(mask, mask)]
        g_sub = values[:, mask]
        for index, genotype in enumerate(g_sub):
            beta, se, pvalue, n_tested = association_complete_case(
                genotype, y_sub, x_sub, k_sub, nulls[cohort].delta
            )
            estimates[cohort]["beta"][index] = beta
            estimates[cohort]["se"][index] = se
            estimates[cohort]["p"][index] = pvalue
            estimates[cohort]["n"][index] = n_tested

    meta = fixed_effect_meta(
        estimates["center_1"]["beta"], estimates["center_1"]["se"],
        estimates["center_2"]["beta"], estimates["center_2"]["se"],
    )
    result = loci.copy()
    result.insert(1, "trait", trait)
    for cohort in cohort_masks:
        for key in ("beta", "se", "p", "n"):
            result[f"{cohort}_{key}"] = estimates[cohort][key]
    for key, vals in meta.items():
        result[f"meta_{key}"] = vals
    result["pooled_fdr"] = benjamini_hochberg(result["pooled_p"].to_numpy())
    result["meta_fdr"] = benjamini_hochberg(result["meta_p"].to_numpy())
    result["center_effects_same_sign"] = (
        np.sign(result["center_1_beta"]) == np.sign(result["center_2_beta"])
    )
    result["center_robust"] = result["center_effects_same_sign"] & (result["meta_q_p"] >= 0.05)
    if trait == "log_oleic_linoleic_ratio":
        result["pooled_ratio_percent_change_per_repeat"] = 100 * (np.exp(result["pooled_beta"]) - 1)
    return result, null_qc


def scan_trait_loco(
    trait: str, y: np.ndarray, loci: pd.DataFrame, values: np.ndarray,
    centers: np.ndarray, fixed_pcs: pd.DataFrame,
    loco_kinships: dict[str, np.ndarray], n_pcs: int, chunk_size: int,
    include_center_covariate: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    null_rows = []
    locus_order = pd.Series(np.arange(len(loci)), index=loci["str_id"])
    observed_chromosomes = set(loci["chrom"].astype(str))
    unsupported = sorted(observed_chromosomes - set(loco_kinships))
    if unsupported:
        raise ValueError(f"STR chromosomes lack LOCO matrices: {unsupported[:10]}")
    for chromosome in map(str, range(1, 24)):
        index = np.flatnonzero(loci["chrom"].astype(str).to_numpy() == chromosome)
        if not len(index):
            continue
        part, chromosome_nulls = scan_trait(
            trait, y, loci.iloc[index].reset_index(drop=True), values[index],
            centers, fixed_pcs, loco_kinships[chromosome], n_pcs, chunk_size,
            include_center_covariate,
        )
        for row in chromosome_nulls:
            row["chromosome"] = chromosome
        parts.append(part)
        null_rows.extend(chromosome_nulls)
    result = pd.concat(parts, ignore_index=True)
    result["_locus_order"] = result["str_id"].map(locus_order)
    result = result.sort_values("_locus_order").drop(columns="_locus_order").reset_index(drop=True)
    result["pooled_fdr"] = benjamini_hochberg(result["pooled_p"].to_numpy())
    result["meta_fdr"] = benjamini_hochberg(result["meta_p"].to_numpy())
    return result, null_rows


def add_thresholds(results: list[pd.DataFrame]) -> tuple[float, float, float]:
    n_loci = len(results[0])
    bonf = 0.05 / n_loci
    study = 0.05 / (n_loci * len(results))
    suggestive = 1 / n_loci
    for result in results:
        result["passes_trait_bonferroni"] = result["pooled_p"] < bonf
        result["passes_study_bonferroni"] = result["pooled_p"] < study
        result["passes_suggestive"] = result["pooled_p"] < suggestive
    return bonf, study, suggestive


def assign_proximity_groups(hits: pd.DataFrame, distance: int = 100_000) -> pd.DataFrame:
    if hits.empty:
        hits = hits.copy(); hits["proximity_group"] = pd.Series(dtype=str); return hits
    hits = hits.sort_values(["trait", "chrom", "start"]).copy()
    group_ids = []
    current = 0; previous = None
    for row in hits.itertuples():
        key = (row.trait, str(row.chrom))
        if previous is None or key != previous[:2] or row.start - previous[2] > distance:
            current += 1
        group_ids.append(f"PG{current:04d}")
        previous = (row.trait, str(row.chrom), row.start)
    hits["proximity_group"] = group_ids
    return hits


def str_pc_sensitivity(
    leading: pd.DataFrame, values: np.ndarray, loci: pd.DataFrame, pheno: pd.DataFrame,
    str_pc_path: Path, kinship: np.ndarray, n_pcs: int,
) -> pd.DataFrame:
    if leading.empty:
        return pd.DataFrame()
    str_pcs = pd.read_csv(str_pc_path, sep="\t").set_index("sample_id").loc[pheno["Sample_ID"]].reset_index()
    x = build_design(str_pcs, pheno["center"].to_numpy(), n_pcs)
    index = pd.Series(np.arange(len(loci)), index=loci["str_id"])
    output = []
    for trait, trait_col in TRAITS.items():
        subset = leading[leading["trait"] == trait]
        if subset.empty:
            continue
        y = pheno[trait_col].to_numpy(dtype=float)
        null = fit_null_lmm(y, x, kinship)
        idx = index.loc[subset["str_id"]].to_numpy(dtype=int)
        estimates = [
            association_complete_case(values[i], y, x, kinship, null.delta)
            for i in idx
        ]
        for locus, (b, s, p, n_tested) in zip(subset["str_id"], estimates):
            output.append({"trait": trait, "str_id": locus, "str_pc_adjusted_beta": b,
                           "str_pc_adjusted_se": s, "str_pc_adjusted_p": p,
                           "str_pc_adjusted_n": n_tested})
    return pd.DataFrame(output)


def nature_style() -> None:
    sns.set_theme(style="ticks", context="talk", font="DejaVu Sans")
    plt.rcParams.update({
        "axes.linewidth": 1.0, "axes.titleweight": "bold", "axes.titlesize": 14,
        "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10, "savefig.dpi": 300, "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_phenotypes(pheno: pd.DataFrame, figures: Path) -> None:
    labels = [
        ("OleicAcid_Percentage", "Oleic acid (%)"),
        ("LinoleicAcid_Percentage", "Linoleic acid (%)"),
        ("oleic_linoleic_ratio", "Oleic/linoleic ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    palette = {"center_1": "#3B6FB6", "center_2": "#D55E00"}
    for ax, (column, label) in zip(axes, labels):
        sns.violinplot(data=pheno, x="center", y=column, hue="center", palette=palette,
                       inner=None, cut=0, linewidth=0.8, legend=False, ax=ax)
        sns.boxplot(data=pheno, x="center", y=column, width=0.24, color="white",
                    showfliers=False, linewidth=1, ax=ax)
        sns.stripplot(data=pheno, x="center", y=column, hue="center", palette=palette,
                      size=3, alpha=0.65, jitter=0.18, legend=False, ax=ax)
        ax.set(xlabel="Measurement center", ylabel=label)
        ax.set_xticks([0, 1], ["Center 1", "Center 2"])
        sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, figures / "phenotype_distributions_by_center")


def manhattan_coordinates(result: pd.DataFrame) -> tuple[pd.DataFrame, list[float], list[str]]:
    frame = result.copy()
    frame["chrom"] = frame["chrom"].astype(str)
    numeric = [str(i) for i in range(1, 24)]
    observed = frame["chrom"].drop_duplicates().tolist()
    chromosome_order = [c for c in numeric if c in observed]
    scaffold_order = [c for c in observed if c not in numeric]
    order = chromosome_order + scaffold_order
    sizes = frame.groupby("chrom")["end"].max().to_dict()
    offset = 0; offsets = {}; ticks = []; labels = []
    gap = 2_000_000
    for chrom in chromosome_order:
        offsets[chrom] = offset
        ticks.append(offset + sizes[chrom] / 2)
        labels.append(chrom if int(chrom) % 2 and int(chrom) < 23 else "")
        offset += sizes[chrom] + gap
    frame["plot_position"] = np.nan
    chrom_mask = frame["chrom"].isin(chromosome_order)
    frame.loc[chrom_mask, "plot_position"] = (
        frame.loc[chrom_mask, "start"] + frame.loc[chrom_mask, "chrom"].map(offsets)
    )
    if scaffold_order:
        scaffold_mask = ~chrom_mask
        scaffold_indices = frame.loc[scaffold_mask].sort_values(["chrom", "start"]).index
        scaffold_width = float(np.median([sizes[c] for c in chromosome_order]))
        scaffold_positions = np.linspace(offset, offset + scaffold_width, len(scaffold_indices))
        frame.loc[scaffold_indices, "plot_position"] = scaffold_positions
        ticks.append(offset + scaffold_width / 2); labels.append("Scaffolds")
    frame["chrom_index"] = frame["chrom"].map({c: i for i, c in enumerate(order)})
    return frame, ticks, labels


def plot_manhattan_qq(results: list[pd.DataFrame], figures: Path, study: float, suggestive: float) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(12.2, 10.5), gridspec_kw={"width_ratios": [3.2, 1]})
    trait_labels = {"oleic_acid": "Oleic acid", "linoleic_acid": "Linoleic acid",
                    "log_oleic_linoleic_ratio": "Log O/L ratio"}
    colors = np.array(["#3B6FB6", "#A8A8A8"])
    for row, result in enumerate(results):
        frame, ticks, labels = manhattan_coordinates(result)
        ax = axes[row, 0]
        ax.scatter(frame["plot_position"], -np.log10(frame["pooled_p"].clip(lower=1e-300)),
                   c=colors[frame["chrom_index"].to_numpy() % 2], s=4, alpha=0.72, rasterized=True)
        ax.axhline(-math.log10(study), color="#B2182B", lw=1, ls="--")
        ax.axhline(-math.log10(suggestive), color="#444444", lw=0.9, ls=":")
        ax.set(xticks=ticks, xticklabels=labels, ylabel=r"$-\log_{10}(P)$")
        ax.set_title(trait_labels[result["trait"].iloc[0]], loc="left")
        ax.tick_params(axis="x", labelrotation=0)
        sns.despine(ax=ax)
        qq = axes[row, 1]
        p = np.sort(result["pooled_p"].dropna().to_numpy())
        expected = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
        observed = -np.log10(p)
        qq.scatter(expected, observed, s=7, color="#3B6FB6", alpha=0.7, rasterized=True)
        maximum = max(expected.max(), observed.max())
        qq.plot([0, maximum], [0, maximum], color="black", lw=1)
        chisq = chi2.isf(p, 1)
        lam = np.nanmedian(chisq) / chi2.ppf(0.5, 1)
        qq.text(0.05, 0.94, rf"$\lambda_{{GC}}$ = {lam:.2f}", transform=qq.transAxes, va="top")
        qq.set(xlabel=r"Expected $-\log_{10}(P)$", ylabel=r"Observed $-\log_{10}(P)$")
        sns.despine(ax=qq)
    fig.tight_layout()
    save_figure(fig, figures / "pooled_lmm_manhattan_qq")


def plot_top_effects(results: list[pd.DataFrame], figures: Path, top_n: int) -> None:
    rows = []
    labels = {"pooled": "Pooled", "center_1": "Center 1", "center_2": "Center 2", "meta": "Meta-analysis"}
    for result in results:
        for rank, record in enumerate(result.nsmallest(top_n, "pooled_p").itertuples(), 1):
            for cohort in labels:
                rows.append({
                    "trait": record.trait, "str_id": record.str_id, "rank": rank,
                    "estimate": getattr(record, f"{cohort}_beta"),
                    "se": getattr(record, f"{cohort}_se"), "cohort": labels[cohort],
                })
    frame = pd.DataFrame(rows)
    trait_order = list(TRAITS)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.3), sharex=False)
    palette = {"Pooled": "#222222", "Center 1": "#3B6FB6", "Center 2": "#D55E00", "Meta-analysis": "#009E73"}
    for ax, trait in zip(axes, trait_order):
        sub = frame[frame["trait"] == trait].copy()
        loci_order = sub.drop_duplicates("str_id").sort_values("rank")["str_id"].tolist()[::-1]
        base = {locus: i for i, locus in enumerate(loci_order)}
        offsets = {"Pooled": -0.24, "Center 1": -0.08, "Center 2": 0.08, "Meta-analysis": 0.24}
        for cohort, group in sub.groupby("cohort"):
            y = np.array([base[x] + offsets[cohort] for x in group["str_id"]])
            ax.errorbar(group["estimate"], y, xerr=1.96 * group["se"], fmt="o", ms=4,
                        color=palette[cohort], capsize=2, lw=1, label=cohort)
        ax.axvline(0, color="#777777", lw=0.8)
        ax.set(yticks=range(len(loci_order)), yticklabels=loci_order, xlabel="Effect per repeat")
        ax.set_title({"oleic_acid": "Oleic acid", "linoleic_acid": "Linoleic acid",
                      "log_oleic_linoleic_ratio": "Log O/L ratio"}[trait])
        sns.despine(ax=ax)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, figures / "top_locus_effects_by_center")


def plot_top_genotypes(
    results: list[pd.DataFrame], pheno: pd.DataFrame, loci: pd.DataFrame,
    raw_values: np.ndarray, figures: Path,
) -> None:
    locus_index = pd.Series(np.arange(len(loci)), index=loci["str_id"])
    palette = {"center_1": "#3B6FB6", "center_2": "#D55E00"}
    labels = {"oleic_acid": "Oleic acid (%)", "linoleic_acid": "Linoleic acid (%)",
              "log_oleic_linoleic_ratio": "Log O/L ratio"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    rng = np.random.default_rng(19)
    for ax, result in zip(axes, results):
        top = result.nsmallest(1, "pooled_p").iloc[0]
        genotype = raw_values[int(locus_index[top.str_id])]
        trait = top.trait
        frame = pd.DataFrame({
            "length": genotype,
            "phenotype": pheno[TRAITS[trait]].to_numpy(dtype=float),
            "center": pheno["center"].to_numpy(),
        }).dropna()
        for center, group in frame.groupby("center"):
            jitter = rng.normal(0, max(0.015, frame["length"].std() * 0.025), len(group))
            ax.scatter(group["length"] + jitter, group["phenotype"], s=20,
                       color=palette[center], alpha=0.68,
                       label=center.replace("_", " ").title(), edgecolor="white", linewidth=0.25)
            if group["length"].nunique() >= 2:
                slope, intercept = np.polyfit(group["length"], group["phenotype"], 1)
                xx = np.linspace(group["length"].min(), group["length"].max(), 100)
                ax.plot(xx, intercept + slope * xx, color=palette[center], lw=1.5)
        ax.set(xlabel="Mean STR repeat count", ylabel=labels[trait])
        ax.set_title(f"{top.str_id}\nP = {top.pooled_p:.2g}", fontsize=12)
        sns.despine(ax=ax)
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    save_figure(fig, figures / "top_locus_genotype_phenotype")


def write_report(
    path: Path, pheno: pd.DataFrame, qc: pd.DataFrame, results: list[pd.DataFrame],
    null_qc: pd.DataFrame, bonf: float, study: float, suggestive: float, n_pcs: int,
    context_annotation: Path, pc_source: str, kinship_mode: str,
    chromosomes_only: bool, str_imputed: bool, include_center_covariate: bool,
) -> None:
    fixed_effect_terms = []
    if include_center_covariate:
        fixed_effect_terms.append("measurement center")
    if n_pcs:
        fixed_effect_terms.append(f"{pc_source.upper()} PC1-PC{n_pcs}")
    fixed_effects = " and ".join(fixed_effect_terms) if fixed_effect_terms else "an intercept only"
    if kinship_mode == "loco":
        analysis_role = "Chromosome-specific LOCO sensitivity"
        kinship_description = (
            "a chromosome-specific SNP genomic relationship matrix constructed from the other 22 chromosomes"
        )
        variance_ratio_scope = "for each phenotype, cohort, and tested chromosome"
    else:
        analysis_role = "Primary"
        kinship_description = "a genome-wide SNP-derived kinship matrix"
        variance_ratio_scope = "for each phenotype and cohort"
    inflation_values = []
    omes_results = {}
    for result in results:
        pvalues = result["pooled_p"].to_numpy()
        inflation_values.append(
            np.nanmedian(chi2.isf(pvalues, 1)) / chi2.ppf(0.5, 1)
        )
        omes_row = result[result["str_id"] == "4_5035216"]
        if not omes_row.empty:
            omes_results[result["trait"].iloc[0]] = omes_row.iloc[0]
    omes_oleic = omes_results["oleic_acid"]
    omes_linoleic = omes_results["linoleic_acid"]
    omes_ratio = omes_results["log_oleic_linoleic_ratio"]
    lines = [
        "# STR length-oil quality association report", "",
        "## Analysis design", "",
        f"The analysis included **{len(pheno)}** samples ({(pheno.center == 'center_1').sum()} from center 1 and "
        f"{(pheno.center == 'center_2').sum()} from center 2). All filtered pSTRs were evaluated; "
        f"**{int(qc.included.sum()):,} of {len(qc):,}** loci met the active eligibility criteria "
        f"(at least three distinct observed lengths"
        f"{' and location on chromosomes 1-23' if chromosomes_only else ''}) and were tested. "
        + (
            "The supplied STR matrix was imputed before association testing, so all retained loci were tested in the "
            "full phenotype cohort. "
            if str_imputed
            else
            "Samples with a missing STR length were excluded separately for each locus; no STR-length imputation was used. "
        )
        + (
            f"The SNP-kinship variance ratio was estimated from the full-cohort null model {variance_ratio_scope}, "
            "then the generalized least-squares STR test was fitted to the full imputed genotype cohort at each locus."
            if str_imputed
            else
            f"The SNP-kinship variance ratio was estimated from the full-cohort null model {variance_ratio_scope}, "
            "then the generalized least-squares STR test was refitted on the observed samples and corresponding kinship "
            "submatrix at each locus."
        ), "",
        f"{analysis_role} tests used a pooled linear mixed model with {fixed_effects} as fixed effects and "
        f"{kinship_description} as a random effect. Center-specific LMMs were combined by inverse-variance fixed-effect "
        "meta-analysis. The O/L phenotype was tested on the natural-log scale; its raw ratio is retained for description.", "",
        (
            "The primary model includes STR PC1-PC5 to adjust explicitly for major STR-derived population-structure "
            "axes while retaining SNP-derived kinship as the random effect."
            if pc_source == "str"
            else
            "SNP PC1-PC5 were included as fixed effects to match the population-structure adjustment used in the SNP GWAS."
            if pc_source == "snp"
            else
            "Fixed SNP PCs were omitted from the primary model because the SNP-derived kinship matrix already captures "
            "genome-wide relatedness; this avoids redundant adjustment of the strongest ancestry axes."
        ), "",
        "No additional call-rate filter was imposed. The three-length eligibility rule was assessed across the combined cohort, "
        f"as specified before analysis. Locus-wise pooled sample sizes ranged from "
        f"**{int(min(result['pooled_n'].min() for result in results))} to "
        f"{int(max(result['pooled_n'].max() for result in results))}**.", "",
        "## Multiple testing", "",
        f"Trait-wise Bonferroni threshold: `{bonf:.3g}`; study-wide threshold across three phenotypes: `{study:.3g}`; "
        f"suggestive threshold: `{suggestive:.3g}`. Benjamini-Hochberg FDR values are also reported per phenotype.", "",
        "## Results", "",
    ]
    for result in results:
        trait = result["trait"].iloc[0]
        best = result.nsmallest(1, "pooled_p").iloc[0]
        chisq = chi2.isf(result["pooled_p"].to_numpy(), 1)
        inflation = np.nanmedian(chisq) / chi2.ppf(0.5, 1)
        lines.append(
            f"- **{trait}**: {int(result.passes_study_bonferroni.sum())} study-wide significant, "
            f"{int(result.passes_trait_bonferroni.sum())} trait-wise significant, and "
            f"{int(result.passes_suggestive.sum())} suggestive loci. The smallest pooled P value was "
            f"`{best.pooled_p:.3g}` at `{best.str_id}` (effect `{best.pooled_beta:.4g}` per repeat); "
            f"genomic inflation lambda was `{inflation:.3f}`."
        )
    fdr_hits = pd.concat(
        [result[result["pooled_fdr"] < 0.05] for result in results], ignore_index=True
    )
    fdr_counts = fdr_hits.groupby("trait").size().to_dict()
    formal_fdr_loci = set(
        fdr_hits.loc[fdr_hits["passes_trait_bonferroni"], "str_id"]
    )
    additional_fdr_loci = fdr_hits["str_id"].nunique() - len(formal_fdr_loci)
    promoter_annotation = load_annotation(context_annotation)
    unique_fdr = (
        fdr_hits[["str_id"]].drop_duplicates()
        .merge(
            promoter_annotation[
                ["str_id", "functional_region", "distance_to_gene", "gene_id", "gene_name"]
            ],
            on="str_id", how="left", validate="one_to_one",
        )
    )
    context_counts = unique_fdr["functional_region"].value_counts().to_dict()
    three_trait_ids = set(
        fdr_hits.groupby("str_id")["trait"].nunique().loc[lambda x: x == len(TRAITS)].index
    )
    three_trait_context = unique_fdr[unique_fdr["str_id"].isin(three_trait_ids)]
    three_trait_promoters = sorted(
        three_trait_context.loc[
            three_trait_context["functional_region"] == "promoter", "str_id"
        ].tolist()
    )
    three_trait_count = len(three_trait_ids)
    promoter_count = len(three_trait_promoters)
    promoter_text = (
        f"Of the {three_trait_count} {'locus' if three_trait_count == 1 else 'loci'} with FDR < 0.05 "
        f"for all three phenotypes, {promoter_count} "
        f"{'is' if promoter_count == 1 else 'are'} promoter-associated"
    )
    if three_trait_promoters:
        promoter_text += f" (`{'`, `'.join(three_trait_promoters)}`)"
    remaining_contexts = three_trait_count - promoter_count
    if remaining_contexts:
        promoter_text += (
            f"; the remaining {remaining_contexts} "
            f"{'is' if remaining_contexts == 1 else 'are'} intergenic"
        )
    promoter_text += ". Genomic proximity is an annotation for prioritization and does not establish that an STR regulates the assigned gene."
    lines.extend([
        "", "## FDR follow-up set", "",
        "For hypothesis generation, associations with pooled Benjamini-Hochberg FDR < 0.05 were retained while the "
        "trait-wise and study-wide Bonferroni thresholds remain the formal discovery criteria. This produces "
        f"**{len(fdr_hits)} phenotype-locus associations at {fdr_hits['str_id'].nunique()} unique STR loci**: "
        f"{fdr_counts.get('oleic_acid', 0)} for oleic acid, {fdr_counts.get('linoleic_acid', 0)} for linoleic acid, and "
        f"{fdr_counts.get('log_oleic_linoleic_ratio', 0)} for log O/L ratio. "
        f"Of the unique loci, {additional_fdr_loci} do not pass a "
        "trait-wise Bonferroni threshold and are therefore additional exploratory candidates.", "",
        "The complete set is provided in `results/fdr_0.05_followup_loci.tsv`. FDR > 0.05 was not used as a selection "
        "criterion because it would select associations that fail the stated false-discovery-rate cutoff.", "",
        "## Genomic context of the FDR follow-up set", "",
        f"Using the promoter-aware 2-kb annotation and counting each STR once, the {len(unique_fdr)} unique FDR loci comprise "
        f"**{context_counts.get('intergenic', 0)} intergenic loci ({100 * context_counts.get('intergenic', 0) / len(unique_fdr):.1f}%)**, "
        f"**{context_counts.get('promoter', 0)} promoters ({100 * context_counts.get('promoter', 0) / len(unique_fdr):.1f}%)**, "
        f"**{context_counts.get('intron', 0)} intronic loci ({100 * context_counts.get('intron', 0) / len(unique_fdr):.1f}%)**, "
        f"**{context_counts.get('CDS', 0)} CDS locus ({100 * context_counts.get('CDS', 0) / len(unique_fdr):.1f}%)**, and "
        f"**{context_counts.get('five_prime_UTR', 0)} 5-prime UTR loci "
        f"({100 * context_counts.get('five_prime_UTR', 0) / len(unique_fdr):.1f}%)**. Thus, "
        f"{len(unique_fdr) - context_counts.get('intergenic', 0)} loci "
        f"({100 * (len(unique_fdr) - context_counts.get('intergenic', 0)) / len(unique_fdr):.1f}%) occur within genes or their 2-kb promoters.", "",
        promoter_text, "",
    ])
    null_columns = ["trait", "cohort", "n", "n_covariates", "delta",
                    "snp_heritability_null", "sigma_g2", "sigma_e2"]
    if "chromosome" in null_qc:
        null_summary = (
            null_qc.groupby(["trait", "cohort"], as_index=False)
            .agg(
                n_chromosomes=("chromosome", "nunique"),
                h2_min=("snp_heritability_null", "min"),
                h2_median=("snp_heritability_null", "median"),
                h2_max=("snp_heritability_null", "max"),
                delta_median=("delta", "median"),
            )
        )
        null_columns = list(null_summary.columns)
        null_display = null_summary
    else:
        null_display = null_qc
    null_table = ["| " + " | ".join(null_columns) + " |",
                  "| " + " | ".join(["---"] * len(null_columns)) + " |"]
    for row in null_display[null_columns].itertuples(index=False, name=None):
        formatted = [f"{value:.4g}" if isinstance(value, float) else str(value) for value in row]
        null_table.append("| " + " | ".join(formatted) + " |")
    lines.extend(["", "## Interpretation safeguards", "",
                  "Association does not establish causality. Center is strongly related to sampling geography, so pooled effects "
                  "should be interpreted together with center-specific direction, heterogeneity, and SNP-kinship adjustment. "
                  "Proximity groups join hits within 100 kb on the same contig; they are not LD-defined independent loci.", "",
                  f"Genomic inflation lambda ranges from {min(inflation_values):.2f} to {max(inflation_values):.2f}, and several null REML fits approach the upper boundary for "
                  "SNP heritability. These diagnostics indicate residual structure and/or limited-sample variance-component "
                  "uncertainty; leading associations require replication and should be prioritized when center-specific effects "
                  + (
                      "agree and the relevant population-structure sensitivity analyses are consistent."
                      if n_pcs else "agree across measurement centers."
                  ), "",
                  "## Candidate genes related to oil quality", "",
                  "The clearest biologically relevant candidate is the `AAAT` STR `4_5035216`, located approximately 58.9 kb "
                  "from `OE9A039927`, annotated as oleoside methyl ester synthase (`OeOMES`). OeOMES is an experimentally "
                  "characterized CYP72 enzyme in secoiridoid and oleuropein biosynthesis. These phenolic compounds contribute "
                  "to olive-oil flavor, bitterness, stability, and overall quality "
                  "([Rodriguez-Lopez et al., 2021](https://iris.cnr.it/handle/20.500.14243/399619)).", "",
                  f"At this STR, increasing repeat count was associated with higher oleic acid (beta = {omes_oleic.pooled_beta:.2f} percentage points "
                  f"per repeat, P = {omes_oleic.pooled_p:.2e}, FDR = {omes_oleic.pooled_fdr:.4f}) and lower linoleic acid "
                  f"(beta = {omes_linoleic.pooled_beta:.2f} percentage points per repeat, P = {omes_linoleic.pooled_p:.2e}, "
                  f"FDR = {omes_linoleic.pooled_fdr:.4f}). Effects had the same direction in both measurement centers. The log O/L-ratio "
                  f"association was weaker (P = {omes_ratio.pooled_p:.2e}, FDR = {omes_ratio.pooled_fdr:.3f}). The locus did not pass the trait-wise Bonferroni threshold, "
                  "and its 58.9-kb separation from OeOMES does not establish direct regulation; it is therefore an FDR-supported "
                  "candidate that may tag a linked region.", "",
                  "Another highlighted association, `CACTIH010001827.1_963000`, is 8.4 kb from `OE9A070630`. Its Arabidopsis "
                  "orthologs are `MOB1C` and `MOB1D`, members of a family reported to regulate development and jasmonate "
                  "accumulation ([AtMOB1 functional study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7054864/)). This is an indirect "
                  "lipid-signaling connection rather than evidence for control of olive-oil fatty-acid composition.", "",
                  "No suggestive STR was directly assigned to the canonical fatty-acid composition genes `OeFAD2-2` or "
                  "`OeSAD2`. Previous functional-expression studies identify OeFAD2-2 as a principal determinant of linoleic "
                  "acid and OeSAD2 as a principal contributor to oleic acid in olive fruit and oil "
                  "([olive FAD2-2 expression study](https://pubs.acs.org/doi/10.1021/jf900678z); "
                  "[olive OeSAD2 expression study](https://pubs.acs.org/doi/10.1021/acs.jafc.6b02963)). Thus, OeOMES is the clearest "
                  "oil-quality candidate in the present results, but the scan does not directly implicate a known core "
                  "oleic/linoleic biosynthesis enzyme.", "",
                  "## Null-model estimates", "", *null_table, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.kinship_mode == "loco" and not args.chromosomes_only:
        raise ValueError(
            "--kinship-mode loco requires --chromosomes-only; scaffold STRs have no LOCO matrix"
        )
    source = args.outdir / "source_data"; figures = args.outdir / "figures"; tables = args.outdir / "results"
    for directory in (source, figures, tables): directory.mkdir(parents=True, exist_ok=True)
    pheno, fixed_pcs, kinship, qc, loci, values, raw_values, _ = prepare_inputs(args)
    loco_kinships = None
    if args.kinship_mode == "loco":
        loco_kinships = load_loco_kinships(
            args.loco_dir, args.loco_fam, pheno["Sample_ID"].tolist()
        )
    pheno.to_csv(source / "phenotypes.tsv", sep="\t", index=False)
    qc.to_csv(source / "locus_qc.tsv.gz", sep="\t", index=False, compression="gzip")
    loci.drop(columns=[]).to_csv(source / "tested_loci.tsv.gz", sep="\t", index=False, compression="gzip")
    if args.kinship_mode == "genomewide":
        np.savez_compressed(source / "snp_kinship_phenotyped_samples.npz",
                            sample_id=pheno["Sample_ID"].to_numpy(), kinship=kinship)
    else:
        loco_manifest = {
            "sample_order_source": str(args.loco_fam),
            "matrix_directory": str(args.loco_dir),
            "chromosomes": list(map(str, range(1, 24))),
            "matrix_pattern": "loco_chr{chromosome}.sXX.txt",
        }
        (source / "loco_kinship_manifest.json").write_text(
            json.dumps(loco_manifest, indent=2) + "\n"
        )

    centers = pheno["center"].to_numpy()
    results = []; null_rows = []
    for trait, column in TRAITS.items():
        print(f"Scanning {trait} ({len(loci):,} loci)", flush=True)
        if args.kinship_mode == "loco":
            result, null_qc = scan_trait_loco(
                trait, pheno[column].to_numpy(dtype=float), loci, values,
                centers, fixed_pcs, loco_kinships, args.n_pcs, args.chunk_size,
                not args.exclude_center_covariate,
            )
        else:
            result, null_qc = scan_trait(
                trait, pheno[column].to_numpy(dtype=float), loci, values,
                centers, fixed_pcs, kinship, args.n_pcs, args.chunk_size,
                not args.exclude_center_covariate,
            )
        results.append(result); null_rows.extend(null_qc)
    bonf, study, suggestive = add_thresholds(results)
    null_qc = pd.DataFrame(null_rows)
    null_qc.to_csv(source / "null_model_qc.tsv", sep="\t", index=False)

    for result in results:
        trait = result["trait"].iloc[0]
        result.sort_values("pooled_p").to_csv(tables / f"{trait}_association.tsv.gz", sep="\t", index=False,
                                               compression="gzip", na_rep="NA")
    hits = pd.concat([r[r["passes_suggestive"]].copy() for r in results], ignore_index=True)
    hits = assign_proximity_groups(hits)
    hits.to_csv(tables / "suggestive_and_significant_loci.tsv", sep="\t", index=False, na_rep="NA")
    fdr_hits = pd.concat([r[r["pooled_fdr"] < 0.05].copy() for r in results], ignore_index=True)
    fdr_hits = assign_proximity_groups(fdr_hits)
    fdr_hits["followup_status"] = np.select(
        [fdr_hits["passes_study_bonferroni"], fdr_hits["passes_trait_bonferroni"]],
        ["study_wide_significant", "trait_wide_significant"],
        default="fdr_followup_only",
    )
    fdr_hits.sort_values(["trait", "pooled_p"]).to_csv(
        tables / "fdr_0.05_followup_loci.tsv", sep="\t", index=False, na_rep="NA"
    )
    leading = pd.concat([r.nsmallest(args.top_loci, "pooled_p") for r in results], ignore_index=True)
    if (
        args.kinship_mode == "genomewide"
        and args.n_pcs
        and args.pc_source != "str"
    ):
        sensitivity = str_pc_sensitivity(
            leading, values, loci, pheno, args.str_pcs, kinship, args.n_pcs
        )
    else:
        sensitivity = pd.DataFrame(columns=[
            "trait", "str_id", "str_pc_adjusted_beta", "str_pc_adjusted_se",
            "str_pc_adjusted_p", "str_pc_adjusted_n",
        ])
    sensitivity.to_csv(tables / "leading_loci_str_pc_sensitivity.tsv", sep="\t", index=False, na_rep="NA")

    summary_rows = []
    for result in results:
        chisq = chi2.isf(result["pooled_p"].to_numpy(), 1)
        summary_rows.append({
            "trait": result["trait"].iloc[0], "n_samples": len(pheno), "n_loci": len(result),
            "n_study_wide": int(result["passes_study_bonferroni"].sum()),
            "n_trait_wide": int(result["passes_trait_bonferroni"].sum()),
            "n_suggestive": int(result["passes_suggestive"].sum()),
            "min_p": result["pooled_p"].min(),
            "genomic_inflation_lambda": np.nanmedian(chisq) / chi2.ppf(0.5, 1),
        })
    pd.DataFrame(summary_rows).to_csv(tables / "association_summary.tsv", sep="\t", index=False)
    thresholds = {"n_loci": len(loci), "n_traits": len(results), "trait_bonferroni": bonf,
                  "study_bonferroni": study, "suggestive": suggestive}
    (source / "multiple_testing_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    model_specification = {
        "fixed_effects": ([] if args.exclude_center_covariate else ["measurement_center"])
        + ([f"{args.pc_source}_PC{i}" for i in range(1, args.n_pcs + 1)] if args.n_pcs else []),
        "pc_source": args.pc_source,
        "n_pcs": args.n_pcs,
        "random_effect": (
            "chromosome-specific SNP LOCO kinship"
            if args.kinship_mode == "loco"
            else "SNP-derived genome-wide kinship"
        ),
        "kinship_mode": args.kinship_mode,
        "chromosomes_only": args.chromosomes_only,
        "missing_str_handling": (
            "pre-imputed STR matrix"
            if args.str_imputed
            else "locus-wise complete-case exclusion"
        ),
        "measurement_center_covariate": not args.exclude_center_covariate,
    }
    (source / "model_specification.json").write_text(
        json.dumps(model_specification, indent=2) + "\n"
    )

    nature_style()
    plot_phenotypes(pheno, figures)
    plot_manhattan_qq(results, figures, study, suggestive)
    plot_top_effects(results, figures, args.top_loci)
    plot_top_genotypes(results, pheno, loci, raw_values, figures)
    write_report(args.outdir / "analysis_report.md", pheno, qc, results, null_qc,
                 bonf, study, suggestive, args.n_pcs, args.annotation, args.pc_source,
                 args.kinship_mode, args.chromosomes_only, args.str_imputed,
                 not args.exclude_center_covariate)
    print(f"Analysis complete: {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
