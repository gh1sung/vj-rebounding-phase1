# VJ Edgecombe Rebounding Study — Phase 1

**Research question:** Does VJ Edgecombe grab more rebounds than his physical profile and team situation predict?

---

## Overview

This is a box-score residual study comparing VJ Edgecombe's rebounding against a cohort of rookie guards with similar size and context. The core idea: if we build a model that predicts rebounding rates from height, minutes, frontcourt teammate quality, and position, the leftover (residual) is the part that physical profile and team context *don't* explain. A large positive residual = above-expectation "knack."

The study runs two models:

- **Phase 1 (script 01):** OLS regression + Random Forest, predicting era-adjusted rebounding z-scores. Controls: height, minutes, frontcourt teammate TRB%, SG indicator.
- **Model B (script 02):** Same framework with NBA Draft Combine wingspan added as a robustness check on the same complete-case sample.

---

## Key Results

**Phase 1 (corrected cohort: 244 players, 2006–2026)**

| Stat | Actual | Predicted | Residual | OLS Pctile | RF Pctile |
|------|-------:|----------:|---------:|-----------:|----------:|
| TRB% | +0.58 σ | +0.11 σ | +0.47 σ | 84th | 83rd |
| ORB% | +0.87 σ | +0.17 σ | +0.70 σ | 90th | 93rd |
| DRB% | +0.32 σ | +0.10 σ | +0.22 σ | 70th | 61st |

**Model B — wingspan robustness (complete-case sample: 160 players)**

| Stat | Model A Residual | Model A Pctile | Model B Residual | Model B Pctile |
|------|----------------:|---------------:|----------------:|---------------:|
| TRB% | +0.376 σ | 80th | +0.418 σ | 82nd |
| ORB% | +0.642 σ | 87th | +0.700 σ | 91st |
| DRB% | +0.158 σ | 66th | +0.185 σ | 68th |

**Bottom line:** VJ's offensive rebounding residual is large, consistent across OLS and RF, and does *not* shrink after controlling for his wingspan (79.5 in). His 3.5-inch wingspan-over-height differential sits at the 45th percentile among combine-measured guards in the sample, so arm length does not explain the signal.

---

## Figures

| Figure | Description |
|--------|-------------|
| `figures/vj_percentile_strip.png` | VJ's residual percentile for TRB%, ORB%, DRB% |
| `figures/model_b_wingspan_comparison.png` | Model A vs Model B residuals side by side |
| `figures/scatter_*.png` | Actual vs predicted z-score scatter for each stat |
| `figures/resid_hist_*.png` | Residual distribution with VJ annotated |
| `figures/diag_*.png` | OLS diagnostic plots (residuals vs fitted & features) |

---

## Repo Structure

```
.
├── source/
│   ├── 01_rebounding_residual.py     # Phase 1 OLS + RF model
│   └── 02_wingspan_robustness.py     # Model B wingspan check
├── data/
│   ├── cohort_rebounding.csv         # Final corrected cohort (244 players)
│   └── cohort_rebounding_wingspan.csv # Cohort merged with combine wingspans
├── results/
│   ├── model_results.csv             # Per-player residuals, Phase 1
│   ├── phase1_verdict.txt            # Written summary, Phase 1
│   ├── model_b_wingspan_results.csv  # Per-player residuals, Model B
│   └── model_b_wingspan_verdict.txt  # Written summary, Model B
├── figures/                          # All output plots (PNG)
├── requirements.txt
└── README.md
```

---

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Phase 1

Pulls data from Basketball Reference (~10–12 min on first run; caches per-season CSVs locally).

```bash
python source/01_rebounding_residual.py
```

To validate the methodology without internet access (uses synthetic data):

```bash
python source/01_rebounding_residual.py --demo
```

### 3. Run Model B

Requires Phase 1 to have run first (`data/cohort_rebounding.csv` must exist). Fetches NBA Draft Combine data via `nba_api`.

```bash
python source/02_wingspan_robustness.py
```

Force-refresh the combine cache:

```bash
python source/02_wingspan_robustness.py --refresh-combine
```

> **Note:** BBRef per-season cache files (`data/bbref_adv_*.csv`) and the combine cache (`data/nba_draft_combine_all_time.csv`) are excluded from version control via `.gitignore` — they are re-fetched automatically on first run.

---

## Important Correction

An earlier version of this study incorrectly identified rookies as "players who first appeared in the dataset in 2006," which misclassified 67 veterans (e.g., Andre Miller, Adrian Griffin) as rookies. The corrected scripts use Basketball Reference's actual `from_year` field. **The valid cohort is 244 players, not 311. Earlier 311-player results are invalid.**

---

## Methods

**Cohort:** Rookie guards (PG/SG), height 6'1"–6'5", ≥500 minutes played, NBA seasons 2006–2026. Rookie year is defined as the player's `from_year` on Basketball Reference.

**Era adjustment:** All rebounding rates (TRB%, ORB%, DRB%) are converted to within-season z-scores relative to all guards in that season. This controls for league-wide rebounding trends.

**Phase 1 model features:**
- `height_inches` — listed height
- `MP` — minutes played
- `frontcourt_trb` — minutes-weighted average TRB% of the top-2 frontcourt teammates (proxy for how much rebounding the bigs "take")
- `is_sg` — SG vs PG indicator

**Model B adds:** `wingspan_inches` from NBA Draft Combine, matched to each player's draft combine year (±2 year tolerance). Both models are fit on the same complete-case sample to isolate the wingspan effect.

**Robustness:** OLS residuals are confirmed against Random Forest out-of-fold residuals (5-fold CV). Model B uses HC3 heteroskedasticity-robust standard errors.

---

## Caveats

- Height only; wingspan is absent from the main Phase 1 model (Model B addresses this).
- `frontcourt_trb` is a season-total proxy, not a true on/off lineup control.
- Box-score rates cannot distinguish contested from uncontested boards.
- Low model R² (0.04–0.08) is expected — the unexplained variance is exactly what the residual captures.

---

## What's Next — Phase 2

Model B rules out arm length as the explanation. Phase 2 uses tracking data to investigate *why* VJ overperforms:

- Rebound chances and conversion rate
- Contested rebound share
- Average rebound distance and ground covered
- Offensive vs defensive chance conversion
- Box-outs (if available)

These distinguish between anticipation/positioning, strength in traffic, and role/opportunity explanations.

---

*Author: thekorean76ers analytics*
