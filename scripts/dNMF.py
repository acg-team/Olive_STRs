#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import argparse

#Example Slurm array (pseudo):
# #SBATCH --array=0-9

# REPEAT_ID=${SLURM_ARRAY_TASK_ID}

# python nmf_cv.py \
#   -f data/str_matrix.tsv \
#   --str-info-file data/str_info.tsv \
#   --period 3 \
#   -r ${REPEAT_ID} \
#   -o results/nmf_period3 \
#   -k 20


def parse_cla():
    parser = argparse.ArgumentParser(
        description="Cross-validate NMF on positive/negative STR genotype matrices."
    )
    parser.add_argument(
        "-f", "--file-path", type=str, required=True,
        help="Path to STR genotype matrix (rows = STRs, cols = samples)."
    )
    parser.add_argument(
        "-r", "--nrepeats", type=int, required=True,
        help="Repeat index (used to set the random seed, usually from Slurm array index)."
    )
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="Output prefix (CSV will be written)."
    )
    parser.add_argument(
        "-k", "--max-k", type=int, default=20,
        help="Maximum K (components) to test (range starts at 3)."
    )
    parser.add_argument(
        "--str-info-file", type=str, default=None,
        help="Optional STR info file with columns 'chr', 'str_id', 'period', 'str_var', 'methods'."
    )
    parser.add_argument(
        "--period", type=int, default=None,
        help="Optional STR period to subset to (e.g. 2, 3, 4...)."
    )
    return parser.parse_args()


def cross_validate_nmf(
    pos_matrix,
    neg_matrix,
    max_k,
    n_repeats=1,
    matching=0.9,
    random_state=42,
    output_path=None
):
    """
    Run NMF on pos/neg matrices for K in [3, max_k).
    In your Slurm setup you typically call this with n_repeats=1
    and change random_state between jobs.

    Returns
    -------
    avecorr_per_k : dict
        K -> {"avercorr", "varcorr", "match", "pos_err", "neg_err"}
        For your use, this is just a single repeat per K.
    """

    K_range = range(2, max_k)
    rng = np.random.default_rng(random_state)
    avecorr_per_k = {k: {} for k in K_range}

    for repeat in range(n_repeats):
        rs = rng.integers(0, 1_000_000)
        print(f"Repeat {repeat + 1}/{n_repeats}, seed={rs}")

        for K in K_range:
            print(f"  K = {K}")

            model_pos = NMF(
                n_components=K,
                init="nndsvd",
                solver="cd",
                max_iter=10000,
                random_state=rs,
            )
            Q_pos = model_pos.fit_transform(pos_matrix)

            model_neg = NMF(
                n_components=K,
                init="nndsvd",
                solver="cd",
                max_iter=10000,
                random_state=rs,
            )
            Q_neg = model_neg.fit_transform(neg_matrix)

            Q_pos /= Q_pos.sum(axis=1, keepdims=True)
            Q_neg /= Q_neg.sum(axis=1, keepdims=True)

            corr_matrix = np.corrcoef(Q_pos.T, Q_neg.T)[:K, K:]
            cost_matrix = -np.abs(corr_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_corr = corr_matrix[row_ind, col_ind]

            avecorr_per_k[K] = {
                "avercorr": total_corr.mean(),
                "varcorr": total_corr.var(),
                "match": int(np.sum(np.abs(total_corr) > matching)),
                "pos_err": float(model_pos.reconstruction_err_),
                "neg_err": float(model_neg.reconstruction_err_),
            }
            
        df_out = pd.DataFrame(avecorr_per_k).T  # K as index
        df_out.index.name = "K"

        out_path = output_path + f"_rep{repeat}.csv"
        df_out.to_csv(out_path)
        print(f"Saved NMF results for repeat {repeat} to: {out_path}")

    #return avecorr_per_k


def split_df(str_file, str_info_file=None, period=None):
    """
    Load STR matrix, optionally subset by STR info & period,
    then standardize and split into positive/negative parts.

    Returns:
      X_pos, X_neg (both samples x STRs)
    """
    # rows = STRs, cols = samples
    str_data = pd.read_csv(str_file, index_col=0, low_memory=False)

    if str_info_file is not None:
        str_info = pd.read_csv(str_info_file).query("str_var > 2").copy()

        # keep only HipSTR calls (3rd field in 'methods' == 1)
        hipstr = np.array([int(i.split("|")[2]) for i in str_info["methods"]])
        str_info = str_info.iloc[hipstr == 1, :]

        str_info["str_uid"] = str_info["chr"] + "_" + str_info["str_id"].astype("str")

        if period is not None:
            keep_uids = str_info.loc[str_info["period"] == period, "str_uid"]
        else:
            keep_uids = str_info["str_uid"]

        # intersect with available rows just in case
        keep_uids = [u for u in keep_uids if u in str_data.index]
        input_data = str_data.loc[keep_uids]
    else:
        input_data = str_data

    # input_data: STRs x samples -> transpose to samples x STRs
    imputed = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(
        input_data.T
    )
    X_data = StandardScaler().fit_transform(imputed)

    X_pos = np.clip(X_data, a_min=0, a_max=None)
    X_neg = np.clip(-X_data, a_min=0, a_max=None)

    return X_pos, X_neg


def main():
    args = parse_cla()

    # each Slurm job: one repeat index, becomes part of the seed
    seed = 10 + args.nrepeats
    print(f"Using random_state={seed} for this job (repeat index {args.nrepeats})")

    X_pos, X_neg = split_df(
        args.file_path,
        str_info_file=args.str_info_file,
        period=args.period,
    )

    cross_validate_nmf(
        pos_matrix=X_pos,
        neg_matrix=X_neg,
        max_k=args.max_k,
        n_repeats=5,          # one repeat per Slurm job
        random_state=seed,
        output_path=args.output_path
    )

    # df_out = pd.DataFrame(res).T  # K as index
    # df_out.index.name = "K"

    # out_path = args.output_path #+ f"_rep{args.nrepeats}.csv"
    # df_out.to_csv(out_path)
    # print(f"Saved NMF results for repeat {args.nrepeats} to: {out_path}")

if __name__ == "__main__":
    main()