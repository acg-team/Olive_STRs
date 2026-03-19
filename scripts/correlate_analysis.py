import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

def load_data(filtered_path, oil_path, sample_col='sample'):
    df_filt = pd.read_csv(filtered_path)
    df_filt = df_filt.set_index('str_id').transpose()
    df_oil = pd.read_csv(oil_path, index_col=0)

    # Intersect sample names by exact match
    shared = df_filt.index.intersection(df_oil.index)
    if len(shared) == 0:
        raise ValueError('No shared sample IDs found between filtered_df and Oil_Data.')

    df_filt_shared = df_filt.loc[shared].copy()
    df_oil_shared = df_oil.loc[shared].copy()

    return df_filt_shared, df_oil_shared


def compute_correlations(df_str, df_oil, min_complete=10):
    results = []
    # Filter str_id columns that are numeric
    numeric_str_cols = df_str.columns
    #[c for c in df_str.columns if pd.api.types.is_numeric_dtype(df_str[c])]
    numeric_oil_cols = df_oil.columns
    #[c for c in df_oil.columns if pd.api.types.is_numeric_dtype(df_oil[c])]

    for str_col in numeric_str_cols:
        x = df_str[str_col]
        for oil_col in numeric_oil_cols:
            y = df_oil[oil_col]
            valid = x.notna() & y.notna()
            if valid.sum() < min_complete:
                continue
            if x[valid].nunique() < 2 or y[valid].nunique() < 2:
                continue
            r, p = pearsonr(x[valid], y[valid]) 
            results.append({
                'str_id': str_col,
                'oil_feature': oil_col,
                'n_samples': int(valid.sum()),
                'r': float(r),
                'p_value': float(p),
            })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Correlate str_id values with oil features across shared samples')
    parser.add_argument('--filtered', default='filtered_df.csv', help='Path to filtered_df.csv')
    parser.add_argument('--oil', default='Oil_Data.csv', help='Path to Oil_Data.csv')
    parser.add_argument('--sample-col', default='sample', help='Sample name column in filtered_df (default: sample)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level for p-values')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum non-missing shared samples per pair')
    parser.add_argument('--output', default='correlation_results.csv', help='Output CSV for all correlation results')
    args = parser.parse_args()

    df_str, df_oil = load_data(args.filtered, args.oil, sample_col=args.sample_col)
    corr_df = compute_correlations(df_str, df_oil, min_complete=args.min_samples)

    if corr_df.empty:
        print('No valid correlation pairs computed.')
        return

    # compute adjusted p-values using Benjamini-Hochberg FDR using statsmodels
    corr_df['adjusted_p'] = multipletests(corr_df['p_value'], method='fdr_bh')[1]       
    corr_df['significant'] = corr_df['adjusted_p'] <= args.alpha

    corr_df.to_csv(args.output, index=False)

    sig = corr_df[corr_df['significant']].sort_values('p_value')
    print(f'Shared samples: {len(df_str)}')
    print(f'Total pairs tested: {len(corr_df)}')
    print(f'Significant pairs at alpha={args.alpha}: {len(sig)}')
    if not sig.empty:
        print('\nTop significant correlations:')
        print(sig.head(30).to_string(index=False))
    else:
        print('No significant pairs found with p <=', args.alpha)

    # also output significant pairs
    sig.to_csv('significant_pairs.csv', index=False)
    
    # plot the most significant pair using seaborn boxplot if any significant pairs exist   
    if not sig.empty:              
        import seaborn as sns
        import matplotlib.pyplot as plt

        top_pair = sig.iloc[0]
        str_col = top_pair['str_id']
        oil_col = top_pair['oil_feature']

        plt.figure(figsize=(8,6))
        sns.boxplot(x=df_str[str_col], y=df_oil[oil_col])
        plt.title(f'Box plot of {str_col} vs {oil_col}\n(r={top_pair["r"]:.2f}, p={top_pair["p_value"]:.2e})')
        plt.xlabel(str_col)
        plt.ylabel(oil_col)
        plt.tight_layout()
        plt.savefig('top_correlation_boxplot.png')
        plt.show()  

if __name__ == '__main__':
    main()
