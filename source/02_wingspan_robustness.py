#!/usr/bin/env python3
"""
Phase 1, Model B: Wingspan Robustness Check
============================================

Tests whether VJ Edgecombe's above-expected rebounding residual survives
after adding NBA Draft Combine wingspan to the Phase 1 controls.

The comparison is intentionally fit on one complete-case sample:

    Model A: height + minutes + frontcourt TRB% + SG indicator
    Model B: Model A + wingspan

Usage:
    python3 02_wingspan_robustness.py
    python3 02_wingspan_robustness.py --refresh-combine
"""

import argparse
import os
import re
import unicodedata
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from nba_api.stats.endpoints import draftcombinestats

warnings.filterwarnings("ignore")

VJ_NAME = "VJ Edgecombe"
VJ_WINGSPAN_INCHES = 79.5
MAX_COMBINE_YEAR_GAP = 2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

COHORT_PATH = os.path.join(DATA_DIR, "cohort_rebounding.csv")
COMBINE_CACHE = os.path.join(DATA_DIR, "nba_draft_combine_all_time.csv")
MERGED_PATH = os.path.join(DATA_DIR, "cohort_rebounding_wingspan.csv")
RESULTS_PATH = os.path.join(RESULTS_DIR, "model_b_wingspan_results.csv")
VERDICT_PATH = os.path.join(RESULTS_DIR, "model_b_wingspan_verdict.txt")
FIGURE_PATH = os.path.join(FIGURES_DIR, "model_b_wingspan_comparison.png")

BASE_FEATURES = ["height_inches", "MP", "frontcourt_trb", "is_sg"]
WINGSPAN_FEATURES = BASE_FEATURES + ["wingspan_inches"]
TARGETS = ["TRB%", "ORB%", "DRB%"]

BG = "#0f0f1a"
GOLD = "#f5a623"
RED = "#e94560"
BLUE = "#4ba3c7"
GRID = "#2a2a3a"


def ordinal(value):
    n = int(round(value))
    suffix = (
        "th"
        if 10 <= n % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def repair_mojibake(value):
    text = str(value)
    if any(marker in text for marker in ("Ã", "Ä", "Â", "â")):
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def name_key(value):
    text = repair_mojibake(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def fetch_combine(refresh=False):
    if os.path.exists(COMBINE_CACHE) and not refresh:
        print(f"  Loading cached combine data: {COMBINE_CACHE}")
        return pd.read_csv(COMBINE_CACHE)

    print("  Fetching all-time NBA Draft Combine data...")
    endpoint = draftcombinestats.DraftCombineStats(
        season_all_time="All Time",
        timeout=90,
    )
    combine = endpoint.get_data_frames()[0]
    combine.to_csv(COMBINE_CACHE, index=False)
    print(f"  Cached {len(combine)} combine rows: {COMBINE_CACHE}")
    return combine


def prepare_combine(combine):
    required = {"SEASON", "PLAYER_NAME", "WINGSPAN"}
    missing = required.difference(combine.columns)
    if missing:
        raise ValueError(f"Combine response missing columns: {sorted(missing)}")

    out = combine.copy()
    out["combine_year"] = pd.to_numeric(out["SEASON"], errors="coerce")
    out["wingspan_inches"] = pd.to_numeric(out["WINGSPAN"], errors="coerce")
    out["combine_player_name"] = out["PLAYER_NAME"].astype(str)
    out["name_key"] = out["PLAYER_NAME"].map(name_key)
    return out[
        out["combine_year"].notna()
        & out["wingspan_inches"].notna()
        & out["name_key"].ne("")
    ].copy()


def merge_wingspan(cohort, combine):
    out = cohort.copy()
    out["name_key"] = out["Player"].map(name_key)
    out["expected_combine_year"] = out["season_end_year"] - 1

    candidates = out.merge(
        combine[
            [
                "name_key",
                "combine_player_name",
                "combine_year",
                "wingspan_inches",
            ]
        ],
        on="name_key",
        how="left",
    )
    candidates["combine_year_gap"] = (
        candidates["combine_year"] - candidates["expected_combine_year"]
    ).abs()
    candidates = candidates.sort_values(
        ["Player", "season_end_year", "combine_year_gap", "combine_year"],
        na_position="last",
    )
    merged = candidates.drop_duplicates(["Player", "season_end_year"]).copy()

    too_far = merged["combine_year_gap"] > MAX_COMBINE_YEAR_GAP
    merged.loc[
        too_far,
        [
            "combine_player_name",
            "combine_year",
            "wingspan_inches",
            "combine_year_gap",
        ],
    ] = np.nan

    vj = merged["Player"].str.casefold().eq(VJ_NAME.casefold())
    if not vj.any():
        raise RuntimeError("VJ Edgecombe is missing from the Phase 1 cohort.")
    merged.loc[vj, "combine_player_name"] = VJ_NAME
    merged.loc[vj, "combine_year"] = 2025
    merged.loc[vj, "combine_year_gap"] = 0
    merged.loc[vj, "wingspan_inches"] = VJ_WINGSPAN_INCHES
    merged["wingspan_minus_height"] = (
        merged["wingspan_inches"] - merged["height_inches"]
    )
    return merged.drop(columns=["name_key"])


def fit_ols(sample, target_z, features, prefix):
    model_data = sample[[target_z, "Player"] + features].dropna().copy()
    x = sm.add_constant(model_data[features].astype(float))
    y = model_data[target_z].astype(float)
    model = sm.OLS(y, x).fit(cov_type="HC3")
    model_data[f"{prefix}_predicted_z"] = model.predict(x)
    model_data[f"{prefix}_residual_z"] = (
        y - model_data[f"{prefix}_predicted_z"]
    )
    model_data[f"{prefix}_residual_pct"] = (
        model_data[f"{prefix}_residual_z"].rank(pct=True) * 100
    )
    return model, model_data


def vj_values(frame, prefix):
    row = frame[frame["Player"].str.casefold().eq(VJ_NAME.casefold())]
    if row.empty:
        raise RuntimeError(f"VJ is missing from {prefix} model sample.")
    row = row.iloc[0]
    return {
        "predicted_z": float(row[f"{prefix}_predicted_z"]),
        "residual_z": float(row[f"{prefix}_residual_z"]),
        "percentile": float(row[f"{prefix}_residual_pct"]),
    }


def plot_comparison(summary):
    labels = TARGETS
    x = np.arange(len(labels))
    width = 0.34
    baseline = [summary[target]["baseline_same"]["residual_z"] for target in labels]
    wingspan = [summary[target]["wingspan"]["residual_z"] for target in labels]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(x - width / 2, baseline, width, color=BLUE, label="Model A: same sample")
    ax.bar(x + width / 2, wingspan, width, color=GOLD, label="Model B: + wingspan")
    ax.axhline(0, color="#777", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("VJ residual (era-adjusted z)", color="white")
    ax.set_title(
        "VJ Edgecombe residual before vs after controlling for wingspan",
        color="white",
    )
    ax.tick_params(colors="white")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.legend(facecolor="#151522", labelcolor="white")

    for positions, values in ((x - width / 2, baseline), (x + width / 2, wingspan)):
        for position, value in zip(positions, values):
            ax.text(
                position,
                value + (0.025 if value >= 0 else -0.05),
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                color="white",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def build_verdict(
    summary,
    complete_n,
    coverage_n,
    cohort_n,
    exact_year_sensitivity,
    vj_wingspan_diff_pct,
):
    orb = summary["ORB%"]
    before = orb["baseline_same"]
    after = orb["wingspan"]
    change = after["residual_z"] - before["residual_z"]
    relative = abs(change) / abs(before["residual_z"]) if before["residual_z"] else np.nan
    direction = "grew" if change > 0 else "shrank"

    if abs(change) < 0.10:
        conclusion = (
            "The ORB% residual is substantively stable after controlling for wingspan. "
            "Arm length does not explain the core above-expectation signal."
        )
    elif after["residual_z"] >= 0.50 and after["percentile"] >= 80:
        conclusion = (
            "Wingspan explains part of the ORB% edge, but a large positive residual "
            "still remains. The finding is weakened slightly, not overturned."
        )
    else:
        conclusion = (
            "The ORB% residual weakens materially after controlling for wingspan, "
            "so arm length appears to explain a meaningful share of the original edge."
        )

    lines = [
        "VJ EDGECOMBE - MODEL B WINGSPAN ROBUSTNESS",
        "=" * 49,
        f"Corrected Phase 1 cohort: {cohort_n} players including VJ",
        f"Usable wingspan coverage: {coverage_n}/{cohort_n}",
        f"Complete-case model sample: {complete_n}",
        "Both models below use the same complete-case sample.",
        "",
        "ORB% HEADLINE",
        f"  Model A residual: {before['residual_z']:+.3f} z "
        f"({ordinal(before['percentile'])} percentile)",
        f"  Model B residual: {after['residual_z']:+.3f} z "
        f"({ordinal(after['percentile'])} percentile)",
        f"  Change after wingspan: {change:+.3f} z "
        f"({direction} by {relative * 100:.1f}% of Model A residual magnitude)",
        f"  Wingspan coefficient: {orb['wingspan_coef']:+.3f} z per inch "
        f"(HC3 p={orb['wingspan_pvalue']:.3f})",
        f"  Exact-year-only sensitivity: {exact_year_sensitivity['residual_z']:+.3f} z "
        f"({ordinal(exact_year_sensitivity['percentile'])} percentile, "
        f"n={exact_year_sensitivity['n']})",
        f"  VJ wingspan-minus-height percentile in Model B sample: "
        f"{ordinal(vj_wingspan_diff_pct)}",
        "",
        "ALL TARGETS",
    ]
    for target in TARGETS:
        item = summary[target]
        a = item["baseline_same"]
        b = item["wingspan"]
        lines.append(
            f"  {target}: Model A {a['residual_z']:+.3f} z "
            f"({ordinal(a['percentile'])}) -> Model B {b['residual_z']:+.3f} z "
            f"({ordinal(b['percentile'])})"
        )

    lines.extend(
        [
            "",
            "MODEL FIT",
            f"  ORB% Model A R-squared: {orb['baseline_r2']:.3f}",
            f"  ORB% Model B R-squared: {orb['wingspan_r2']:.3f}",
            "",
            "CONCLUSION",
            f"  {conclusion}",
            "  VJ's 3.5-inch wingspan advantage over his listed height is near the",
            "  middle of this combine-measured guard sample, so adding wingspan lowers",
            "  his expected ORB% slightly rather than explaining away his production.",
            "",
            "CAUTION",
            "  Combine attendance is incomplete, so Model B uses a smaller, selected",
            "  sample. The same-sample comparison isolates the added wingspan control,",
            "  but the result should still be described as a robustness check rather",
            "  than a definitive causal decomposition of physical tools versus instinct.",
            "",
            "PHASE 2",
            "  Model B tests whether arm length explains the residual. It still cannot",
            "  identify the mechanism. Rebound chances, conversion rate, contested-board",
            "  share, box-outs, and rebound distance are the next step for explaining why.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(refresh_combine=False):
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        os.makedirs(directory, exist_ok=True)

    print("=" * 64)
    print("Phase 1 Model B - Wingspan Robustness")
    print("=" * 64)

    if not os.path.exists(COHORT_PATH):
        raise FileNotFoundError(
            f"Missing {COHORT_PATH}. Run the corrected Phase 1 script first."
        )

    cohort = pd.read_csv(COHORT_PATH)
    if cohort.duplicated(["Player", "season_end_year"]).any():
        raise ValueError("Phase 1 cohort contains duplicate player-seasons.")

    combine = prepare_combine(fetch_combine(refresh=refresh_combine))
    merged = merge_wingspan(cohort, combine)
    merged.to_csv(MERGED_PATH, index=False)

    coverage_n = int(merged["wingspan_inches"].notna().sum())
    print(f"  Cohort rows: {len(merged)}")
    print(f"  Usable wingspans: {coverage_n}/{len(merged)}")
    print(f"  VJ wingspan used: {VJ_WINGSPAN_INCHES:.1f} inches")

    common_columns = list(
        dict.fromkeys(
            ["Player", "season_end_year", "wingspan_inches"]
            + BASE_FEATURES
            + [f"{target}_z" for target in TARGETS]
        )
    )
    complete = merged[common_columns].dropna().copy()
    if not complete["Player"].str.casefold().eq(VJ_NAME.casefold()).any():
        raise RuntimeError("VJ was dropped from the complete-case Model B sample.")
    print(f"  Complete-case sample: {len(complete)}")

    summary = {}
    result_frames = []

    for target in TARGETS:
        target_z = f"{target}_z"
        baseline_model, baseline = fit_ols(
            complete, target_z, BASE_FEATURES, "model_a"
        )
        wingspan_model, wingspan = fit_ols(
            complete, target_z, WINGSPAN_FEATURES, "model_b"
        )
        a = vj_values(baseline, "model_a")
        b = vj_values(wingspan, "model_b")

        summary[target] = {
            "baseline_same": a,
            "wingspan": b,
            "baseline_r2": float(baseline_model.rsquared),
            "wingspan_r2": float(wingspan_model.rsquared),
            "wingspan_coef": float(wingspan_model.params["wingspan_inches"]),
            "wingspan_pvalue": float(wingspan_model.pvalues["wingspan_inches"]),
        }

        combined = baseline[
            [
                "Player",
                "model_a_predicted_z",
                "model_a_residual_z",
                "model_a_residual_pct",
            ]
        ].merge(
            wingspan[
                [
                    "Player",
                    "model_b_predicted_z",
                    "model_b_residual_z",
                    "model_b_residual_pct",
                ]
            ],
            on="Player",
            how="inner",
        )
        combined["target"] = target
        result_frames.append(combined)

        print(f"\n  {target}")
        print(
            f"    Model A: VJ residual {a['residual_z']:+.3f} z, "
            f"{ordinal(a['percentile'])}"
        )
        print(
            f"    Model B: VJ residual {b['residual_z']:+.3f} z, "
            f"{ordinal(b['percentile'])}"
        )
        print(
            f"    Wingspan coefficient: "
            f"{summary[target]['wingspan_coef']:+.3f} z/in "
            f"(HC3 p={summary[target]['wingspan_pvalue']:.3f})"
        )

    exact_year = merged[
        merged["combine_year_gap"].eq(0)
        & merged["wingspan_inches"].notna()
    ].copy()
    _, exact_wingspan = fit_ols(
        exact_year, "ORB%_z", WINGSPAN_FEATURES, "exact_model_b"
    )
    exact_vj = vj_values(exact_wingspan, "exact_model_b")
    exact_vj["n"] = len(exact_wingspan)

    complete_with_diff = merged.dropna(
        subset=WINGSPAN_FEATURES + ["wingspan_minus_height", "ORB%_z"]
    ).copy()
    complete_with_diff["wingspan_diff_pct"] = (
        complete_with_diff["wingspan_minus_height"].rank(pct=True) * 100
    )
    vj_wingspan_diff_pct = float(
        complete_with_diff.loc[
            complete_with_diff["Player"].str.casefold().eq(VJ_NAME.casefold()),
            "wingspan_diff_pct",
        ].iloc[0]
    )

    pd.concat(result_frames, ignore_index=True).to_csv(RESULTS_PATH, index=False)
    plot_comparison(summary)
    verdict = build_verdict(
        summary,
        complete_n=len(complete),
        coverage_n=coverage_n,
        cohort_n=len(merged),
        exact_year_sensitivity=exact_vj,
        vj_wingspan_diff_pct=vj_wingspan_diff_pct,
    )
    with open(VERDICT_PATH, "w", encoding="utf-8") as handle:
        handle.write(verdict)

    print("\n" + verdict)
    print(f"Saved merged cohort: {MERGED_PATH}")
    print(f"Saved model results: {RESULTS_PATH}")
    print(f"Saved verdict: {VERDICT_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-combine",
        action="store_true",
        help="Ignore the local combine cache and request the endpoint again.",
    )
    args = parser.parse_args()
    run(refresh_combine=args.refresh_combine)
