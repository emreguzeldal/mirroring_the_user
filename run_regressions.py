"""
Regression analysis for the 'Mirroring the User' study.

Reads the feature parquet produced by `run_production.py` and estimates
the framing-register relationship under four progressively stronger
specifications:

    Specification A - Pooled OLS with HC3 standard errors
    Specification B - Topic fixed effects (TF-IDF + KMeans clustering)
    Specification C - Conversation fixed effects (cluster-robust SEs)
    Specification D - Conversation FE + lagged register (autoregressive control)

Also reports:
    - Variance decomposition (between-conversation, between-topic, within)
    - Per-component regression of each register feature on framing under FE
    - Topic clustering diagnostics (silhouette over candidate K)

Usage
-----
    python run_regressions.py --in /path/to/multiturn_features.parquet --out ./results
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# -------------------- Topic clustering --------------------

def cluster_topics(prompts: pd.Series,
                   k_candidates=(20, 30, 40, 50, 75, 100),
                   seed: int = 42):
    """TF-IDF + KMeans topic clustering with silhouette-based K selection.

    Returns
    -------
    labels : np.ndarray
        Topic label for each prompt
    best_k : int
        Selected number of topics
    diag : pd.DataFrame
        Diagnostic table with silhouette and inertia for each candidate K
    """
    vec = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        sublinear_tf=True,
    )
    X = vec.fit_transform(prompts.tolist())

    rows, best_k, best_s, best_labels = [], None, -np.inf, None
    rng = np.random.RandomState(seed)
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)
        s = silhouette_score(X[idx], labels[idx], metric="cosine")
        rows.append({"k": k, "silhouette": s, "inertia": km.inertia_})
        if s > best_s:
            best_s, best_k, best_labels = s, k, labels
    return best_labels, best_k, pd.DataFrame(rows)


# -------------------- Variance decomposition --------------------

def variance_decomposition(df: pd.DataFrame, var: str, group: str) -> dict:
    """Decompose variance of `var` into between-group and within-group components."""
    grand = df[var].mean()
    between = df.groupby(group)[var].mean().sub(grand).pow(2)
    sizes = df.groupby(group).size()
    ss_b = float((between * sizes).sum())
    ss_t = float(((df[var] - grand) ** 2).sum())
    return {
        "var": var,
        "ss_between": ss_b,
        "ss_within": ss_t - ss_b,
        "ss_total": ss_t,
        "share_between": ss_b / ss_t if ss_t > 0 else np.nan,
    }


def demean(df: pd.DataFrame, vars_: list, group: str) -> pd.DataFrame:
    """Within-transform (de-mean) variables by group. Adds new columns with '_wc' suffix."""
    for v in vars_:
        df[f"{v}_wc"] = df[v] - df.groupby(group)[v].transform("mean")
    return df


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.inp)
    df = df.sort_values(["conversation_hash", "turn_idx"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from {df['conversation_hash'].nunique():,} conversations")

    # Topic clustering
    print("\n=== Clustering topics ===")
    labels, best_k, diag = cluster_topics(df["prompt"])
    df["topic"] = labels
    diag.to_csv(out / "topic_diag.csv", index=False)
    print(f"  Best k = {best_k}")
    print(diag.to_string(index=False))

    # Variance decomposition
    print("\n=== Variance decomposition ===")
    vd_rows = []
    for v in ["framing_score", "register_score"]:
        for grp in ["conversation_hash", "topic"]:
            d = variance_decomposition(df, v, grp)
            d["group"] = grp
            vd_rows.append(d)
            print(f"  {v:18s} grouped by {grp:20s} share_between={d['share_between']:.3f}")
    pd.DataFrame(vd_rows).to_csv(out / "variance_decomposition.csv", index=False)

    # Specifications
    df["model_gpt4"] = (df["model_version"] == "gpt4").astype(int)

    specs = {}

    # SPEC A: Pooled OLS with HC3 standard errors
    print("\n=== SPEC A: Pooled OLS (HC3) ===")
    mA = smf.ols(
        "register_score ~ framing_score * C(model_version) "
        "+ log_p_tokens + log_r_tokens + turn_idx",
        data=df,
    ).fit(cov_type="HC3")
    specs["A_pooled"] = mA
    print(f"  framing_score: b={mA.params['framing_score']:+.4f}  "
          f"SE={mA.bse['framing_score']:.4f}  p={mA.pvalues['framing_score']:.4g}")

    # SPEC B: Topic FE
    print("\n=== SPEC B: Topic FE (HC3) ===")
    mB = smf.ols(
        "register_score ~ framing_score * C(model_version) "
        "+ log_p_tokens + log_r_tokens + turn_idx + C(topic)",
        data=df,
    ).fit(cov_type="HC3")
    specs["B_topicFE"] = mB
    print(f"  framing_score: b={mB.params['framing_score']:+.4f}  "
          f"SE={mB.bse['framing_score']:.4f}  p={mB.pvalues['framing_score']:.4g}")

    # SPEC C: Conversation FE (within-transformed, cluster-robust SEs)
    print("\n=== SPEC C: Conversation FE (cluster-robust) ===")
    df = demean(df, ["framing_score", "register_score", "log_p_tokens",
                     "log_r_tokens", "turn_idx"], "conversation_hash")
    mC = smf.ols(
        "register_score_wc ~ framing_score_wc + log_p_tokens_wc "
        "+ log_r_tokens_wc + turn_idx_wc",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["conversation_hash"]})
    specs["C_convFE"] = mC
    print(f"  framing_score: b={mC.params['framing_score_wc']:+.4f}  "
          f"SE={mC.bse['framing_score_wc']:.4f}  p={mC.pvalues['framing_score_wc']:.4g}")

    # SPEC D: Conversation FE + lagged register
    print("\n=== SPEC D: Conversation FE + lagged register ===")
    df_lag = df.dropna(subset=["lag_register"]).copy()
    df_lag = demean(df_lag, ["framing_score", "register_score", "log_p_tokens",
                             "log_r_tokens", "turn_idx", "lag_register"],
                    "conversation_hash")
    mD = smf.ols(
        "register_score_wc ~ framing_score_wc + lag_register_wc "
        "+ log_p_tokens_wc + log_r_tokens_wc + turn_idx_wc",
        data=df_lag,
    ).fit(cov_type="cluster", cov_kwds={"groups": df_lag["conversation_hash"]})
    specs["D_convFE_lag"] = mD
    print(f"  framing_score: b={mD.params['framing_score_wc']:+.4f}  "
          f"SE={mD.bse['framing_score_wc']:.4f}  p={mD.pvalues['framing_score_wc']:.4g}")
    print(f"  lag_register : b={mD.params['lag_register_wc']:+.4f}  "
          f"SE={mD.bse['lag_register_wc']:.4f}  p={mD.pvalues['lag_register_wc']:.4g}")

    # Headline summary
    print("\n" + "=" * 78)
    print("=== HEADLINE: framing -> register across all specifications ===")
    print("=" * 78)
    rows = []
    for name, m in specs.items():
        coef_name = ("framing_score_wc"
                     if "framing_score_wc" in m.params.index
                     else "framing_score")
        rows.append({
            "spec": name,
            "framing_b": m.params[coef_name],
            "framing_se": m.bse[coef_name],
            "framing_p": m.pvalues[coef_name],
            "R2": m.rsquared,
            "N": int(m.nobs),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(out / "regression_summary.csv", index=False)

    # Save full coefficient tables for each specification
    for name, m in specs.items():
        ct = pd.DataFrame({
            "coef": m.params,
            "se": m.bse,
            "t": m.tvalues,
            "p": m.pvalues,
            "ci_lo": m.conf_int()[0],
            "ci_hi": m.conf_int()[1],
        })
        ct.to_csv(out / f"coef_{name}.csv")

    # Per-component register decomposition under conversation FE
    print("\n=== Per-register-component framing effect (Spec C: conversation FE) ===")
    comp_rows = []
    for r_feat in ["r_first_person_sg", "r_second_person", "r_affect_words",
                   "r_hedges", "r_empath_social", "r_analytic_raw",
                   "r_politeness", "r_relational_verbs"]:
        if r_feat not in df.columns:
            continue
        df = demean(df, [r_feat], "conversation_hash")
        m = smf.ols(
            f"{r_feat}_wc ~ framing_score_wc + log_p_tokens_wc "
            f"+ log_r_tokens_wc + turn_idx_wc",
            data=df,
        ).fit(cov_type="cluster", cov_kwds={"groups": df["conversation_hash"]})
        comp_rows.append({
            "component": r_feat,
            "b": m.params["framing_score_wc"],
            "se": m.bse["framing_score_wc"],
            "p": m.pvalues["framing_score_wc"],
        })
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    comp_df.to_csv(out / "per_component_framing.csv", index=False)

    print(f"\nAll outputs written to: {out}")


if __name__ == "__main__":
    main()
