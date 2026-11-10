import os
import numpy as np
import pandas as pd
from scipy.stats import norm, cauchy
from statsmodels.stats.multitest import multipletests

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")


def wald_pvals_se(beta_vec, se_vec):
    """
    计算 Wald 检验的 P 值
    """
    beta_vec = np.array(beta_vec, dtype=float)
    se_vec = np.array(se_vec, dtype=float)

    if len(beta_vec) != len(se_vec):
        raise ValueError("beta_vec and se_vec must have same length.")

    p = np.full_like(beta_vec, np.nan)

    valid_mask = np.isfinite(beta_vec) & np.isfinite(se_vec) & (se_vec > 0)

    if np.any(valid_mask):
        z = beta_vec[valid_mask] / se_vec[valid_mask]
        p_vals = 2 * norm.sf(np.abs(z))
        p[valid_mask] = p_vals

    p = np.clip(p, 1e-15, 1 - 1e-15)

    return p


def cauchy_combine_mats(P_list, weights=None):
    """
    使用 Cauchy 组合规则合并 P 值矩阵 (ACAT)
    """
    M = len(P_list)
    if M < 1:
        raise ValueError("P_list must contain at least one matrix")

    if weights is None:
        weights = np.ones(M) / M
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)

    nr, nc = P_list[0].shape
    for i in range(1, M):
        if P_list[i].shape != (nr, nc):
            raise ValueError("All matrices in P_list must have the same shape")

    Tmat = np.zeros((nr, nc))

    for m in range(M):
        P = np.array(P_list[m])
        P[np.isnan(P)] = 1.0
        P = np.clip(P, 1e-15, 1 - 1e-15)

        Tmat += weights[m] * np.tan((0.5 - P) * np.pi)

    P_comb = 1.0 - cauchy.cdf(Tmat)

    P_comb = np.clip(P_comb, 1e-15, 1 - 1e-15)

    return P_comb



G = 4740
K = 15

P_list = {}
beta_list = {}
data_mode = "gau"
pat_names = ["gau2", "cos2", "gau1", "cos1", "linear"]

for pat_id, pat_name in enumerate(pat_names):
    file_path = f"./yc_results/data_{data_mode}/CTSV_Fed_8_G_1000_L1_2e-07_Round1000_h_DataMode_{data_mode}_BasisMode_{pat_name}_Inference_FIXED.csv"

    try:
        beta_raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue

    if len(beta_raw) != G * 2 * K:
        raise ValueError(f"Rows in {pat_name} do not match G*2*K")

    beta_list[pat_name] = beta_raw


    df = beta_raw.copy()

    df['row_idx'] = np.arange(len(df))

    df['gene_id'] = (df['row_idx'] // (2 * K)) + 1

    df['pos'] = df['row_idx'] % (2 * K)

    df['ct'] = (df['pos'] // 2) + 1

    df['basis'] = np.where(df['pos'] % 2 == 0, "h1", "h2")

    df['colname'] = "ct" + df['ct'].astype(str) + "_" + df['basis']

    beta_hat_df = df.pivot(index='gene_id', columns='colname', values='Learned_Beta')
    expected_cols = [f"ct{k}_{b}" for k in range(1, K + 1) for b in ["h1", "h2"]]
    beta_hat = beta_hat_df[expected_cols].values

    se_hat_df = df.pivot(index='gene_id', columns='colname', values='Std_Error')
    se_hat = se_hat_df[expected_cols].values

    P_mat = np.zeros((G, 2 * K))

    for g in range(G):
        P_mat[g, :] = wald_pvals_se(beta_hat[g, :], se_hat[g, :])

    P_list[pat_name] = P_mat

P_list_ordered = [P_list[name] for name in pat_names]

P_comb = cauchy_combine_mats(P_list_ordered)

p_vector = P_comb.flatten()

_, q_vector, _, _ = multipletests(p_vector, method='fdr_bh')

Q_val_comb = q_vector.reshape(P_comb.shape)


P_save = np.zeros((G, K))
Q_save = np.zeros((G, K))

for k in range(K):
    col_idx1 = 2 * k
    col_idx2 = 2 * k + 1

    p_subset = P_comb[:, [col_idx1, col_idx2]]
    q_subset = Q_val_comb[:, [col_idx1, col_idx2]]

    P_save[:, k] = np.min(p_subset, axis=1)
    Q_save[:, k] = np.min(q_subset, axis=1)

gene_names = [f"Gene{i}" for i in range(1, G + 1)]
ct_names = [f"CT{i}" for i in range(1, K + 1)]

Q_save_df = pd.DataFrame(Q_save, index=gene_names, columns=ct_names)

alpha = 0.01
q_gene = Q_save_df.min(axis=1)
sv_genes = q_gene[q_gene < alpha].index.tolist()

print(f"Number of SV genes: {len(sv_genes)}")

true_sv_genes = set([f"Gene{i}" for i in range(101, 701)])
detected_sv_genes = set(sv_genes)

TP = len(detected_sv_genes.intersection(true_sv_genes))
FP = len(detected_sv_genes - true_sv_genes)
FN = len(true_sv_genes - detected_sv_genes)
TN = G - TP - FP - FN

sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
accuracy = (TP + TN) / (TP + FP + FN + TN)

print(f"sensitivity: {sensitivity:.4f} precision: {precision:.4f} "
      f"specificity: {specificity:.4f} accuracy: {accuracy:.4f}")