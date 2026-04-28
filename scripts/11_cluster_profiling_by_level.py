"""Same k-means idea as script 10 but you pick automation_level (default L2). Output prefix is level lowercased."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import baseline_common as bc
import narrative_utils as nu

_MPL = bc.PROJECT_ROOT / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# columns used to characterize each cluster's dominant crash context
PROFILE_COLS = [
    "Roadway Type",
    "Crash With",
    "SV Pre-Crash Movement",
    "CP Pre-Crash Movement",
]

# module-level storage for global nav flag means, used to compute per-cluster lift
_GLOBAL_NAV_MEANS: dict[str, float] = {}


# impute, encode, and standardize features into a numpy array for k-means
def build_cluster_X(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list]:
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    cat_cols = [f for f in features if f not in num_cols]
    X = df[features].copy()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("Unknown")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(float)
    sc = StandardScaler()
    X.iloc[:, :] = sc.fit_transform(X)
    return X.values, list(X.columns)


# try k=3..8 and pick the value with the highest silhouette score
def pick_k(X: np.ndarray, k_range=range(3, 9), seed: int = 42) -> tuple[int, list]:
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=seed)
        scores.append((k, s))
        print(f"   k={k}  silhouette={s:.4f}")
    best_k = max(scores, key=lambda t: t[1])[0]
    print(f"\n   Best k = {best_k}")
    return best_k, scores


# summarize one cluster: size, severe rate, automation level, and top category per profile column
def profile_cluster(df_cluster: pd.DataFrame, cluster_id: int, n_total: int, level: str) -> dict:
    pct = len(df_cluster) / n_total * 100
    row = {
        "cluster": cluster_id,
        "n": len(df_cluster),
        "pct_of_level": round(pct, 1),
        "severe_rate": round(df_cluster["severe"].mean() * 100, 1),
        "automation_level": level,
    }
    for col in PROFILE_COLS:
        if col in df_cluster.columns and not df_cluster[col].dropna().empty:
            top_val = df_cluster[col].value_counts().index[0]
            top_pct = df_cluster[col].value_counts().iloc[0] / len(df_cluster) * 100
            row[f"top_{col.replace(' ', '_')}"] = f"{top_val} ({top_pct:.0f}%)"
        else:
            row[f"top_{col.replace(' ', '_')}"] = "N/A"
    return row


# assign a human-readable scenario label based on crash profile and dominant nav flag lift
def label_cluster(profile: dict, df_cluster: pd.DataFrame) -> str:
    roadway = profile.get("top_Roadway_Type", "")
    sv_move = profile.get("top_SV_Pre-Crash_Movement", "")
    crash_with = profile.get("top_Crash_With", "")
    severe_rate = profile.get("severe_rate", 0)

    if "Animal" in crash_with:
        return f"Animal strike — {severe_rate:.0f}% severe"
    if "Fixed Object" in crash_with or "Other Fixed Object" in crash_with:
        return f"AV hit fixed object — {severe_rate:.0f}% severe"

    nav_cols = [c for c in df_cluster.columns if c.startswith("nav_")]
    if nav_cols and _GLOBAL_NAV_MEANS:
        cluster_means = df_cluster[nav_cols].mean()
        lift = cluster_means - pd.Series(_GLOBAL_NAV_MEANS).reindex(nav_cols, fill_value=0)
        top_lift_flag = lift.idxmax()
        top_lift_val = lift.max()
        top_nav = top_lift_flag if top_lift_val > 0.05 else None
    elif nav_cols:
        nav_means = df_cluster[nav_cols].mean().sort_values(ascending=False)
        top_nav = nav_means.index[0] if nav_means.iloc[0] > 0.05 else None
    else:
        top_nav = None

    nav_label_map = {
        "nav_av_stopped": "AV stopped — rear-end risk",
        "nav_other_struck_av": "Other vehicle struck AV",
        "nav_av_struck_other": "AV struck other party",
        "nav_rear_approach": "Rear-approach / rear-end",
        "nav_at_intersection": "Intersection conflict",
        "nav_on_highway": "Highway driving",
        "nav_left_turn": "Left-turn conflict",
        "nav_lane_change": "Lane-change conflict",
        "nav_av_changing_lanes": "AV changing lanes",
        "nav_av_turning": "AV turning",
        "nav_av_reversing": "AV reversing",
        "nav_in_parking_lot": "Parking-lot scenario",
        "nav_vulnerable_user": "Pedestrian / cyclist involved",
        "nav_emergency_vehicle": "Emergency vehicle present",
        "nav_large_vehicle": "Large vehicle (bus / truck)",
        "nav_av_disengaged": "AV disengaged / takeover",
        "nav_hazard_lights": "Hazard lights / disabled AV",
        "nav_double_parked": "Double-parked obstruction",
        "nav_speed_mentioned": "Speed cited in narrative",
        "nav_minor_damage_lang": "Minor-damage language",
        "nav_av_moving": "AV moving at impact",
    }

    if "Highway" in roadway or "Freeway" in roadway:
        ctx = "Highway"
    elif "Intersection" in roadway:
        ctx = "Intersection"
    elif "Parking" in roadway:
        ctx = "Parking lot"
    else:
        ctx = "Street"

    if "Stopped" in sv_move or "Parked" in sv_move:
        sv = "AV stopped"
    elif "Left Turn" in sv_move or "Right Turn" in sv_move:
        sv = "AV turning"
    elif "Backing" in sv_move:
        sv = "AV reversing"
    elif "Lane" in sv_move or "Changing" in sv_move:
        sv = "AV changing lanes"
    else:
        sv = "AV moving"

    if top_nav and top_nav in nav_label_map:
        core = nav_label_map[top_nav]
    else:
        core = f"Typical {ctx.lower()} crash ({sv})"

    return f"{core} — {severe_rate:.0f}% severe"


# produce six-panel cluster diagnostics figure for the chosen automation level
def make_cluster_figure(
    df_slice: pd.DataFrame,
    profiles: list[dict],
    k: int,
    sil_scores: list,
    level: str,
    out_png: Path,
) -> None:
    fig_dir = out_png.parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    ax_sil = fig.add_subplot(gs[0, 0])
    ks = [t[0] for t in sil_scores]
    sils = [t[1] for t in sil_scores]
    best_k = ks[sils.index(max(sils))]
    ax_sil.plot(ks, sils, marker="o", color="#4C72B0")
    ax_sil.axvline(best_k, color="#C44E52", linestyle="--", alpha=0.7)
    ax_sil.set_title(f"Silhouette score vs k ({level})", fontweight="semibold")
    ax_sil.set_xlabel("k")
    ax_sil.set_ylabel("Silhouette score")

    ax_sz = fig.add_subplot(gs[0, 1])
    labels = [f"C{p['cluster']}" for p in profiles]
    sizes = [p["n"] for p in profiles]
    ax_sz.bar(labels, sizes, color="#4C72B0", alpha=0.85)
    ax_sz.set_title(f"Cluster sizes ({level})", fontweight="semibold")
    ax_sz.set_xlabel("Cluster")
    ax_sz.set_ylabel("Incidents")

    ax_sv = fig.add_subplot(gs[0, 2])
    rates = [p["severe_rate"] for p in profiles]
    colors_bar = ["#C44E52" if r >= 50 else "#4C72B0" for r in rates]
    ax_sv.bar(labels, rates, color=colors_bar, alpha=0.85)
    ax_sv.axhline(
        df_slice["severe"].mean() * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Overall {level} avg",
    )
    ax_sv.set_title("Severe rate per cluster", fontweight="semibold")
    ax_sv.set_xlabel("Cluster")
    ax_sv.set_ylabel("% severe")
    ax_sv.legend(fontsize=8)

    ax_rd = fig.add_subplot(gs[1, :2])
    road_col = "Roadway Type"
    if road_col in df_slice.columns:
        ct = df_slice.groupby(["cluster", road_col]).size().unstack(fill_value=0)
        ct.plot(kind="bar", ax=ax_rd, stacked=True, alpha=0.85, legend=True)
        ax_rd.set_title("Roadway Type distribution per cluster", fontweight="semibold")
        ax_rd.set_xlabel("Cluster")
        ax_rd.set_ylabel("Incidents")
        ax_rd.tick_params(axis="x", rotation=0)
        ax_rd.legend(fontsize=7, loc="upper right", ncol=2)

    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    tbl_data = [[f"C{p['cluster']}", p.get("scenario_label", "—")[:45]] for p in profiles]
    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Cluster", "Scenario"],
        cellLoc="left",
        loc="center",
        colWidths=[0.15, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax_tbl.set_title("Cluster scenario labels", fontweight="semibold", pad=12)

    fig.suptitle(f"{level} crash scenario clusters (k={k})", fontsize=14, fontweight="bold")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   Figure: {out_png}")


# accept --level argument to select which automation level to cluster
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-means clustering by automation_level (symmetry with script 10).")
    p.add_argument(
        "--level",
        type=str,
        default="L2",
        help="Value of automation_level to cluster (default: L2). Must exist in cleaned data.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = args.level.strip()
    prefix = level.lower().replace(" ", "_")

    print("=" * 70)
    print(f"{level} crash scenario clustering (by-level symmetry script)")
    print("=" * 70)

    # load data with narrative flags and validate the requested automation level exists
    df_known = bc.load_known()
    df_known = nu.attach_narrative_flags(df_known)

    levels_present = df_known["automation_level"].dropna().unique().tolist()
    if level not in levels_present:
        print(f"   ERROR: automation_level '{level}' not in data. Found: {levels_present}")
        raise SystemExit(1)

    slice_df = df_known[df_known["automation_level"] == level].copy()
    print(f"   {level} incidents: {len(slice_df)}  |  severe rate: {slice_df['severe'].mean()*100:.1f}%")

    if len(slice_df) < 30:
        print("   ERROR: too few rows for stable k-means / silhouette (need at least ~30).")
        raise SystemExit(1)

    tab_feats = [f for f in bc.context_features(slice_df) if f != "automation_level"]
    features = tab_feats + nu.NAV_FEATURES
    features = [f for f in features if f in slice_df.columns]

    print(f"\n   Building feature matrix ({len(features)} features)…")
    X, _ = build_cluster_X(slice_df, features)

    print("\n   Selecting k via silhouette score:")
    best_k, sil_scores = pick_k(X)

    # fit final k-means with best k and assign cluster labels to each incident
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    slice_df = slice_df.copy()
    slice_df["cluster"] = km.fit_predict(X)

    _GLOBAL_NAV_MEANS.clear()
    nav_cols_present = [c for c in slice_df.columns if c.startswith("nav_")]
    _GLOBAL_NAV_MEANS.update(slice_df[nav_cols_present].mean().to_dict())

    print("\n" + "=" * 70)
    print("Cluster profiles")
    print("=" * 70)

    # profile and label each cluster, printing nav flag lift signals
    profiles = []
    for cid in sorted(slice_df["cluster"].unique()):
        sub = slice_df[slice_df["cluster"] == cid]
        pr = profile_cluster(sub, cid, len(slice_df), level)
        pr["scenario_label"] = label_cluster(pr, sub)
        profiles.append(pr)

        nav_cols_sub = [c for c in sub.columns if c.startswith("nav_")]
        if nav_cols_sub and _GLOBAL_NAV_MEANS:
            c_means = sub[nav_cols_sub].mean()
            lift_s = c_means - pd.Series(_GLOBAL_NAV_MEANS).reindex(nav_cols_sub, fill_value=0)
            top3 = lift_s.nlargest(3)
        else:
            top3 = pd.Series(dtype=float)

        print(f"\n  Cluster {cid} — {pr['n']} incidents ({pr['pct_of_level']:.1f}% of {level})")
        print(f"    Severe rate : {pr['severe_rate']:.1f}%")
        print(f"    Scenario    : {pr['scenario_label']}")
        if not top3.empty:
            print(f"    Top nav lifts: " + ", ".join(f"{k}+{v:.2f}" for k, v in top3.items()))
        for col in PROFILE_COLS:
            key = f"top_{col.replace(' ', '_')}"
            print(f"    {col:<30} {pr.get(key, 'N/A')}")

    bc.CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)

    # save cluster assignments and summary CSV, then generate the diagnostics figure
    out_csv = bc.CLUSTERING_DIR / f"{prefix}_cluster_assignments.csv"
    save_cols = ["cluster"] + [c for c in slice_df.columns if c != "cluster"]
    slice_df[save_cols].to_csv(out_csv, index=False)
    print(f"\n   Saved cluster assignments: {out_csv}")

    summary_df = pd.DataFrame(profiles)
    summary_csv = bc.CLUSTERING_DIR / f"{prefix}_cluster_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"   Saved cluster summary    : {summary_csv}")

    out_png = bc.CLUSTERING_DIR / f"{prefix}_kmeans_cluster_profiles_figure.png"
    make_cluster_figure(slice_df, profiles, best_k, sil_scores, level, out_png)


if __name__ == "__main__":
    main()
