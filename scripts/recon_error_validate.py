"""Validate SAE reconstruction error (FVU) vs the rarity composite against
independent phylogenetic-novelty labels (orphan orthogroup / missing mouse ortholog).

Reads data/recon_error_sample.csv (produced by recon_error_prototype.py).
Run with KMP_DUPLICATE_LIB_OK=TRUE.
"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

df = pd.read_csv("data/recon_error_sample.csv")
df["no_og"] = df.orthogroup.fillna("").str.strip().eq("")        # no orthogroup at all
for c in ["singleton_og", "missing_mouse"]:
    df[c] = df[c].astype(bool)
labels = ["no_og", "singleton_og", "missing_mouse"]
preds = {"FVU_mean": df.fvu_mean, "FVU_max": df.fvu_max,
         "rarity_composite": df.composite, "n_res(length)": df.n_res}

print("AUROC (>0.5 => higher score predicts novelty):")
for name, p in preds.items():
    print(f"  {name:18s}", {l: round(roc_auc_score(df[l], p), 3) for l in labels})

print("\nSpearman rho (predictor vs label):")
for name, p in preds.items():
    print(f"  {name:18s}",
          {l: round(spearmanr(p, df[l].astype(int)).correlation, 3) for l in labels})

print("\nIncremental AUROC over log-length (logistic; correct direction learned):")
for l in ["no_og", "missing_mouse"]:
    y = df[l].astype(int).values
    Xl = np.log(df.n_res.values).reshape(-1, 1)
    base = roc_auc_score(y, LogisticRegression(max_iter=1000).fit(Xl, y).predict_proba(Xl)[:, 1])
    def add(col):
        X = np.column_stack([np.log(df.n_res), df[col]]); X = (X - X.mean(0)) / X.std(0)
        return roc_auc_score(y, LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1])
    print(f"  {l:14s} length={base:.3f}  +FVU={add('fvu_mean'):.3f}  +composite={add('composite'):.3f}")
