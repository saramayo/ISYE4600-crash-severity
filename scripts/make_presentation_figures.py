"""Build PNGs under Presentation/figures/. Needs Cleaned + Modeling outputs from earlier scripts."""
from __future__ import annotations

from pathlib import Path
import os

_ROOT = Path(__file__).resolve().parent.parent
_MPL = _ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import baseline_common as bc

# set up figure output directory and slide map path
PROJECT_ROOT = _ROOT
FIG_DIR = PROJECT_ROOT / "Presentation" / "figures"
MAP_PATH = PROJECT_ROOT / "Presentation" / "SLIDE_FIGURES.txt"

FIG_KW = dict(dpi=150, bbox_inches="tight")
WIDE = (12.0, 6.75)


# save current matplotlib figure as a PNG to FIG_DIR
def _save(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / f"{name}.png"
    plt.savefig(p, **FIG_KW)
    plt.close()
    return p


# apply consistent title font, tick size, and spine style to an axis
def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    ax.tick_params(axis="both", labelsize=11)
    sns.despine(ax=ax)


# fig 01 — outcome distribution bar and OR-rule component breakdown among severe incidents
def fig_severity_and_label_breakdown(df: pd.DataFrame) -> None:
    k = df[df["severity_known"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=WIDE)

    vc = k["severe"].value_counts().reindex([0, 1], fill_value=0)
    axes[0].bar(["Non-severe", "Severe"], [vc[0], vc[1]], color=["#4C72B0", "#C44E52"])
    axes[0].set_ylabel("Incidents (labeled)", fontsize=12)
    style_axes(axes[0], "Outcome distribution (severity-known rows)")

    sev = k[k["severe"] == 1]
    parts = []
    for col, lab in [
        ("injury_flag", "Injury ≥ moderate"),
        ("airbag_flag", "Airbag deployed"),
        ("towed_flag", "Vehicle towed"),
    ]:
        if col in sev.columns:
            s = sev[col].dropna()
            parts.append((lab, float(s.mean()) if len(s) else 0.0))
    labs = [p[0] for p in parts]
    vals = [p[1] * 100 for p in parts]
    axes[1].barh(labs, vals, color="#55A868")
    axes[1].set_xlabel("% of severe incidents with signal = 1 (each can overlap)", fontsize=11)
    style_axes(axes[1], "Severity label OR-rule — components among severe")

    plt.tight_layout()
    _save("01_severity_outcome_and_label_components")


# fig 02 — visual schematic of the OR-rule used to build the binary severity label
def fig_label_rule_schematic() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.3, 1.2, 2.2, 1.0, "Injury\n≥ Moderate"),
        (3.0, 1.2, 2.2, 1.0, "Airbag\ndeployed"),
        (5.7, 1.2, 2.2, 1.0, "Vehicle\ntowed"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor="#E8E8E8", edgecolor="#333"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=12, fontweight="semibold")

    ax.text(8.3, 1.7, "OR", fontsize=16, fontweight="bold")
    ax.add_patch(mpatches.FancyBboxPatch((8.9, 1.0), 0.9, 1.4, boxstyle="round,pad=0.05", facecolor="#C44E52", edgecolor="#333"))
    ax.text(9.35, 1.7, "Severe\n= 1", ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    ax.text(5, 2.55, "Binary severity label (after consolidating to one row per incident)", ha="center", fontsize=13, fontweight="semibold")
    plt.tight_layout()
    _save("02_severity_label_rule_schematic")


# fig 03 — bar chart showing archived train vs current-era test split sizes
def fig_temporal_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    labs = ["Train\n(archived)", "Test\n(current)"]
    vals = [len(train_df), len(test_df)]
    c = ["#8172B3", "#CCB974"]
    ax.bar(labs, vals, color=c, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"n = {v:,}", ha="center", fontsize=12, fontweight="semibold")
    ax.set_ylabel("Labeled incidents", fontsize=12)
    style_axes(ax, "Temporal validation — train on past reports, test on current era")
    plt.tight_layout()
    _save("03_temporal_train_test_split")


# fig 04 — severe rate by era × automation level, highlighting reporting bias
def fig_reporting_bias_strata(df: pd.DataFrame) -> None:
    k = df[df["severity_known"] == 1].copy()
    g = k.groupby(["era", "automation_level"], observed=False).agg(
        n=("severe", "size"),
        severe_rate=("severe", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(len(g))
    ax.bar(x, g["severe_rate"] * 100, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['era']}\n{r['automation_level']}" for _, r in g.iterrows()], fontsize=10)
    ax.set_ylabel("Severe rate (%)", fontsize=12)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(i, r["severe_rate"] * 100 + 1, f"n={int(r['n'])}", ha="center", fontsize=9, color="#333")
    style_axes(ax, "Reporting strata — severe rate differs by era and automation level")
    plt.tight_layout()
    _save("04_reporting_bias_severe_rate_by_stratum")


# fig 05 — pooled logistic regression precision/recall/F1 bar chart
def fig_baseline_metrics(br: pd.DataFrame) -> None:
    main = br[br["model"] == "Logistic Regression"].copy()
    if main.empty:
        return
    row = main.iloc[0]
    fig, ax = plt.subplots(figsize=WIDE)
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    vals = [row[m] for m in metrics]
    ax.bar(x, vals, width=0.5, color="#4C72B0", label="Logistic regression")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    style_axes(ax, "Logistic regression baseline (pooled, current-era test)")
    plt.tight_layout()
    _save("05_baseline_metrics_lr")


# fig 06 — confusion matrix heatmap for the pooled LR baseline
def fig_confusion_heatmaps(br: pd.DataFrame) -> None:
    sub = br[br["model"] == "Logistic Regression"]
    if sub.empty:
        return

    r = sub.iloc[0]
    tn, fp, fn, tp = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        annot_kws={"size": 13},
    )
    ax.set_title("Logistic regression", fontsize=13, fontweight="semibold")
    plt.suptitle("Confusion matrix (current-era test)", fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()
    _save("06_confusion_matrices")


# fig 07 — side-by-side metric bars for ADS and L2 stratified models
def fig_stratified_models(strat: pd.DataFrame) -> None:
    if strat.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=False)
    metrics = ["precision", "recall", "f1"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    labels = ["Precision", "Recall", "F1"]

    for ax, (_, row) in zip(axes, strat.iterrows()):
        level = row["automation_level"]
        vals = [row[m] for m in metrics]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="semibold")
        feat = str(row.get("feature_set", "")).replace("_", " ")
        thr = float(row.get("threshold", 0.5))
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{level} model\n({feat}, threshold={thr:.2f})",
                     fontsize=13, fontweight="semibold")
        ax.set_ylabel("Score")
        auc = row["roc_auc"]
        fn = row["fn_rate"]
        ax.text(0.98, 0.97, f"AUC={auc:.3f}\nFN-rate={fn:.1%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5"))

    fig.suptitle("Separate LR models per automation level — primary results",
                 fontsize=14, fontweight="semibold")
    plt.tight_layout()
    _save("07_stratified_model_results_ads_l2")


# fig 14 — side-by-side confusion matrix heatmaps for ADS and L2 stratified models
def fig_stratified_confusion_matrices(strat: pd.DataFrame) -> None:
    if strat.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, (_, row) in zip(axes, strat.iterrows()):
        level = row["automation_level"]
        cm = np.array([[int(row["TN"]), int(row["FP"])],
                       [int(row["FN"]), int(row["TP"])]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Non-severe", "Pred Severe"],
                    yticklabels=["True Non-severe", "True Severe"],
                    annot_kws={"size": 13})
        feat = str(row.get("feature_set", "")).replace("_", " ")
        ax.set_title(f"{level} — {feat}", fontsize=13, fontweight="semibold")
    fig.suptitle("Confusion matrices — stratified LR models (current-era test)",
                 fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()
    _save("14_stratified_confusion_matrices_ads_l2")


# fig 15 — grouped bars comparing pooled LR vs stratified ADS model on precision/recall/F1
def fig_ads_improvement(br: pd.DataFrame, strat: pd.DataFrame) -> None:
    pooled_ads = br[br["model"] == "LR — test [ADS]"]
    strat_ads = strat[strat["automation_level"] == "ADS"]
    if pooled_ads.empty or strat_ads.empty:
        return

    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    pooled_vals = [float(pooled_ads.iloc[0][m]) for m in metrics]
    strat_vals = [float(strat_ads.iloc[0][m]) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, pooled_vals, width=w, label="Pooled LR (ADS slice)", color="#4C72B0", alpha=0.8)
    ax.bar(x + w / 2, strat_vals, width=w, label="Stratified ADS model\n(narrative flags, tuned threshold)",
           color="#DD8452", alpha=0.9)
    for i, (pv, sv) in enumerate(zip(pooled_vals, strat_vals)):
        ax.text(i - w / 2, pv + 0.02, f"{pv:.2f}", ha="center", fontsize=10)
        ax.text(i + w / 2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    style_axes(ax, "ADS model: pooled LR failure → stratified fix")
    plt.tight_layout()
    _save("15_ads_pooled_vs_stratified_improvement")


# fig 08 — horizontal bar breakdown of roadway type and crash partner in false negatives
def fig_false_negatives(fn: pd.DataFrame) -> None:
    if fn.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, col, title in [
        (axes[0], "Roadway Type", "False negatives — roadway type"),
        (axes[1], "Crash With", "False negatives — crash partner (Crash With)"),
    ]:
        if col not in fn.columns:
            continue
        s = fn[col].value_counts().head(8)
        s.plot(kind="barh", ax=ax, color="#C44E52")
        style_axes(ax, title)
    plt.suptitle("Where logistic regression misses severe crashes (FN analysis)", fontsize=14, fontweight="semibold", y=1.02)
    plt.tight_layout()
    _save("08_false_negative_contexts")


# fig 09 — top LR odds ratios as horizontal bars, colored by direction
def fig_odds_ratios(coef_path: Path, top_n: int = 12) -> None:
    coef = pd.read_csv(coef_path)
    coef = coef.reindex(coef["coefficient"].abs().sort_values(ascending=False).index).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(coef["coefficient"].values >= 0, "#C44E52", "#4C72B0")
    ax.barh(coef["feature"][::-1], coef["odds_ratio"][::-1], color=colors[::-1])
    ax.axvline(1.0, color="#333", linestyle="--", linewidth=1)
    ax.set_xlabel("Odds ratio (vs reference category / unit)", fontsize=11)
    style_axes(ax, "Interpretable drivers — top logistic regression odds ratios")
    plt.tight_layout()
    _save("09_top_odds_ratios_lr")


# fig 10 — silhouette curve and PCA scatter for exploratory k-means clustering
def fig_clustering(df_known: pd.DataFrame) -> None:
    feats = bc.context_features(df_known)
    X_raw, _, _ = bc.prepare_X(df_known, feats)
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    X = X_raw.copy()
    if num_cols:
        X[num_cols] = StandardScaler().fit_transform(X[num_cols])
    X = X.fillna(0)
    if len(X) > 2500:
        X = X.sample(2500, random_state=42)

    ks = list(range(2, 9))
    sil = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        sil.append(silhouette_score(X, lab) if len(np.unique(lab)) > 1 else np.nan)

    best_k = int(ks[int(np.nanargmax(sil))])

    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    axes[0].plot(ks, sil, marker="o", color="#8172B3")
    axes[0].set_xlabel("k (clusters)", fontsize=12)
    axes[0].set_ylabel("Silhouette score", fontsize=12)
    style_axes(axes[0], "Cluster quality vs k (K-means, scaled context features)")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X.values)
    sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", alpha=0.5, s=12)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(sc, ax=axes[1], label="Cluster")
    style_axes(axes[1], f"PCA projection — K-means with k={best_k} (exploratory)")

    plt.tight_layout()
    _save("10_clustering_silhouette_and_pca")


# fig 11 — interpretation talking-point slide with key findings and caveats
def fig_interpretation_summary() -> None:
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axis("off")
    lines = [
        "Interpretation for the system (talk track + bullets)",
        "",
        "• Prioritize validation where LR flags higher odds (see odds-ratio figure).",
        "• Address ADS vs L2 gap: model behavior differs by automation level on current-era test.",
        "• False negatives cluster on streets / intersections / passenger-car strikes — targeted scenarios.",
        "• Caveat: SGO reporting is not a census; use for prioritization signals, not population rates.",
    ]
    y = 0.92
    for i, line in enumerate(lines):
        fs = 16 if i == 0 else 13
        wt = "bold" if i == 0 else "normal"
        ax.text(0.05, y - i * 0.09, line, fontsize=fs, fontweight=wt, transform=ax.transAxes, family="sans-serif")
    plt.tight_layout()
    _save("11_interpretation_for_system_talking_slide")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # load data and generate EDA, label, and temporal split figures
    df = pd.read_csv(bc.DATA_PATH, low_memory=False)
    df_known = df[df["severity_known"] == 1].copy()
    train_df, test_df = bc.split_train_test(df_known)

    fig_severity_and_label_breakdown(df)
    fig_label_rule_schematic()
    fig_temporal_split(train_df, test_df)
    fig_reporting_bias_strata(df)

    # load baseline results and generate baseline metrics + confusion matrix figures
    br = pd.read_csv(PROJECT_ROOT / "Modeling" / "baselines" / "baseline_results.csv")
    fig_baseline_metrics(br)
    fig_confusion_heatmaps(br)

    # load stratified results and generate stratified model comparison figures
    strat_path = PROJECT_ROOT / "Modeling" / "logistic_regression" / "lr_stratified_by_level_results.csv"
    strat = pd.read_csv(strat_path) if strat_path.exists() else pd.DataFrame()
    fig_stratified_models(strat)
    fig_stratified_confusion_matrices(strat)
    fig_ads_improvement(br, strat)

    # generate FN analysis and odds ratio figures if prior model outputs exist
    fn_path = PROJECT_ROOT / "Modeling" / "logistic_regression" / "false_negatives.csv"
    if fn_path.exists():
        fig_false_negatives(pd.read_csv(fn_path))
    coef_path = PROJECT_ROOT / "Modeling" / "logistic_regression" / "lr_ads_coefficients.csv"
    if coef_path.exists():
        fig_odds_ratios(coef_path)
    fig_clustering(df_known)
    fig_interpretation_summary()

    # write slide map listing all generated figures in suggested presentation order
    paths = sorted(FIG_DIR.glob("*.png"))
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Presentation figures (PNG) — map to slides",
        "Generated by scripts/make_presentation_figures.py",
        "",
    ]
    for p in paths:
        lines.append(f"- {p.name}")
    lines.extend(
        [
            "",
            "Suggested slide order:",
            "  02 — Label rule (methods)",
            "  01 — Outcome + OR components (methods / data)",
            "  03 — Temporal split (methods)",
            "  04 — Reporting bias strata (limitations / motivation for stratified approach)",
            "  05–06 — Logistic baseline + confusion matrix (context / what failed)",
            "  15 — ADS pooled vs stratified: before/after improvement story",
            "  07 — Primary results: separate ADS and L2 models (main result)",
            "  14 — Stratified confusion matrices (ADS and L2 side by side)",
            "  08 — False negatives (error analysis)",
            "  09 — Odds ratios / narrative coefficients (interpretation)",
            "  10 — Clustering (methods / exploratory results)",
            "  11 — Interpretation for system (discussion)",
            "  13 — Pooled vs stratified bar comparison (appendix / backup)",
        ]
    )
    MAP_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(paths)} figures to {FIG_DIR}")
    print(f"Slide map: {MAP_PATH}")


if __name__ == "__main__":
    main()
