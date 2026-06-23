#!/usr/bin/env python3
"""
Phase 1: VJ Edgecombe Expected-Rebounding Residual Study
=========================================================

Research question: Does VJ Edgecombe grab MORE rebounds than his
physical profile and team situation predict?

Approach: OLS regression predicting TRB%/ORB%/DRB% (era-adjusted
z-scores) from height, minutes, frontcourt teammate TRB%, and position.
Residual = actual - predicted = the "knack" above what size+context explain.

Run with real internet access to pull BBRef data.
Use --demo flag for synthetic validation without internet.

Usage:
    python3 01_rebounding_residual.py              # real data (BBRef)
    python3 01_rebounding_residual.py --demo       # synthetic demo

Author: thekorean76ers analytics — Phase 1
"""

import os, sys, time, warnings, argparse, requests
from io import StringIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  PARAMETERS
# ─────────────────────────────────────────────────────────────
HEIGHT_MIN_INCHES = 73       # 6'1"
HEIGHT_MAX_INCHES = 77       # 6'5"
MIN_MINUTES       = 500
SEASON_START      = 2006
SEASON_END        = 2026
VJ_NAME           = "VJ Edgecombe"

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

BBREF_HDR = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

GUARD_1ST   = {"PG", "SG", "G"}
FC_1ST      = {"C", "PF", "F"}
FEATURES    = ["height_inches", "MP", "frontcourt_trb", "is_sg"]

# Visual style
BG = "#0f0f1a"; GOLD = "#f5a623"; RED = "#e94560"; BLUE = "#1a5276"; GRID = "#1e1e2e"

def _dark_ax(ax):
    ax.set_facecolor(BG); ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.title.set_color("white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

def ordinal(n):
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def normalize_bbref_advanced(df):
    """Normalize BBRef's current headers to the short names used below."""
    if "Tm" not in df.columns and "Team" in df.columns:
        df = df.rename(columns={"Team": "Tm"})
    return df

# ─────────────────────────────────────────────────────────────
# 1.  BBRef DATA FETCHING
# ─────────────────────────────────────────────────────────────
def fetch_bbref_advanced(year):
    cache = os.path.join(DATA_DIR, f"bbref_adv_{year}.csv")
    if os.path.exists(cache):
        return normalize_bbref_advanced(pd.read_csv(cache, dtype=str))
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    try:
        r   = requests.get(url, headers=BBREF_HDR, timeout=30); r.raise_for_status()
        df  = pd.read_html(StringIO(r.text), header=0)[0]
        df  = normalize_bbref_advanced(df)
        df  = df[df["Player"] != "Player"].reset_index(drop=True)
        df["season_end_year"] = str(year)
        df.to_csv(cache, index=False)
        print(f"  ✓ advanced {year}  ({len(df)} rows)"); time.sleep(3.5)
        return df
    except Exception as e:
        print(f"  ✗ advanced {year}: {e}"); return None


def fetch_bbref_heights():
    cache = os.path.join(DATA_DIR, "bbref_player_heights.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        try:
            r   = requests.get(f"https://www.basketball-reference.com/players/{letter}/",
                               headers=BBREF_HDR, timeout=30); r.raise_for_status()
            dfs = pd.read_html(StringIO(r.text), header=0)
            if dfs: rows.append(dfs[0])
            time.sleep(3.5)
        except Exception as e:
            print(f"  ✗ players/{letter}: {e}")
    if not rows: return None
    out = pd.concat(rows, ignore_index=True)
    rn  = {}
    for c in out.columns:
        cl = c.lower().strip()
        if cl == "player": rn[c] = "player_name"
        elif cl == "from":  rn[c] = "from_year"
        elif cl in ("ht","height"): rn[c] = "height_str"
    out.rename(columns=rn, inplace=True)
    def p(h):
        try: ft,i = str(h).strip().split("-"); return int(ft)*12+int(i)
        except: return np.nan
    if "height_str" in out.columns:
        out["height_inches"] = out["height_str"].apply(p)
    out.to_csv(cache, index=False)
    print(f"  ✓ heights cached ({len(out)} players)")
    return out


def load_all_advanced():
    frames = []
    for y in range(SEASON_START, SEASON_END + 1):
        df = fetch_bbref_advanced(y)
        if df is not None: frames.append(df)
    if not frames: raise RuntimeError("No BBRef data loaded. Check internet / use --demo.")
    return pd.concat(frames, ignore_index=True)

# ─────────────────────────────────────────────────────────────
# 2.  SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────
def make_synthetic_cohort(n=265, seed=42):
    """
    Realistic distributions for rookie guards 6'1"-6'5", calibrated
    to known NBA ranges.  VJ is added with real confirmed values but
    NaN for rate stats (those need BBRef).
    """
    rng = np.random.default_rng(seed)
    ht  = rng.uniform(73, 77, n)
    mp  = np.clip(rng.lognormal(7.1, 0.45, n), 500, 2900)
    fc  = rng.normal(20, 3.5, n)
    sg  = rng.binomial(1, 0.57, n)
    yr  = rng.integers(SEASON_START, SEASON_END, n)

    def make_stat(base, h_coef, fc_coef, mp_coef, sg_coef, noise_std, lo, hi):
        v = (base + h_coef*(ht-75) + fc_coef*(fc-20)
             + mp_coef*(mp-1200) + sg_coef*sg + rng.normal(0,noise_std,n))
        return np.clip(v, lo, hi)

    trb = make_stat(5.0,  0.25, -0.09, 0.0008, -0.40, 1.6, 1.5, 15.0)
    orb = make_stat(2.6,  0.12, -0.04, 0.0003, -0.25, 0.7, 0.5,  8.0)
    drb = make_stat(8.2,  0.30, -0.10, 0.0010, -0.35, 2.0, 2.0, 20.0)

    df = pd.DataFrame({
        "Player": [f"SynPlayer_{i:03d}" for i in range(n)],
        "Pos": np.where(sg, "SG", "PG"),
        "height_inches": ht, "season_end_year": yr.astype(int),
        "Tm": "SYN", "MP": mp,
        "TRB%": trb, "ORB%": orb, "DRB%": drb,
        "frontcourt_trb": fc, "is_sg": sg,
    })
    # Append VJ with confirmed real values; rate stats = NaN until BBRef run
    vj = {
        "Player": VJ_NAME, "Pos": "SG",
        "height_inches": 76.0,      # 6'4" confirmed
        "season_end_year": 2026,
        "Tm": "PHI", "MP": 2623.0,  # confirmed: 75G × 35.0 MPG
        "TRB%": np.nan,             # ← FILL FROM BBREF (est ≈ 9.0%)
        "ORB%": np.nan,             # ← FILL FROM BBREF
        "DRB%": np.nan,             # ← FILL FROM BBREF
        "frontcourt_trb": 20.0,     # PHI had Embiid + Lob City bigs
        "is_sg": 1,
    }
    return pd.concat([df, pd.DataFrame([vj])], ignore_index=True)

# ─────────────────────────────────────────────────────────────
# 3.  COHORT FILTERING
# ─────────────────────────────────────────────────────────────
def _is_guard(pos):
    if pd.isna(pos): return False
    return str(pos).upper().split("-")[0].strip() in GUARD_1ST

def _is_fc(pos):
    if pd.isna(pos): return False
    return str(pos).upper().split("-")[0].strip() in FC_1ST

def _resolve_traded(grp):
    tm = grp["Tm"].astype(str).str.upper()
    aggregate = tm.eq("TOT") | tm.str.match(r"^\d+TM$")
    return grp[aggregate] if aggregate.any() else grp

def build_cohort(adv_all, heights):
    for col in ["MP","TRB%","ORB%","DRB%","season_end_year"]:
        if col in adv_all.columns:
            adv_all[col] = pd.to_numeric(adv_all[col], errors="coerce")
    clean = (adv_all.groupby(["Player","season_end_year"], group_keys=False)
             .apply(_resolve_traded).reset_index(drop=True))

    clean["name_key"]    = clean["Player"].str.strip().str.lower()
    heights["name_key"]  = heights["player_name"].str.strip().str.lower()
    merged = clean.merge(
        heights[["name_key","height_inches","from_year"]].drop_duplicates("name_key"),
        on="name_key", how="left")

    merged["rookie_year"] = pd.to_numeric(merged["from_year"], errors="coerce")
    merged = merged[merged["rookie_year"] >= SEASON_START]
    cohort = merged[merged["season_end_year"] == merged["rookie_year"]].copy()
    cohort = cohort[
        cohort["Pos"].apply(_is_guard)
        & cohort["MP"].ge(MIN_MINUTES)
        & cohort["height_inches"].ge(HEIGHT_MIN_INCHES)
        & cohort["height_inches"].le(HEIGHT_MAX_INCHES)
    ].copy()
    cohort["is_sg"] = cohort["Pos"].str.upper().str.startswith("SG").astype(int)
    print(f"  Cohort: {len(cohort)} players"); return cohort.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 4.  FRONTCOURT TRB CONTROL
# ─────────────────────────────────────────────────────────────
def compute_frontcourt_trb(adv_all, cohort):
    a = adv_all.copy()
    for c in ["MP","TRB%","season_end_year"]:
        a[c] = pd.to_numeric(a.get(c, np.nan), errors="coerce")
    a = (a.groupby(["Player","season_end_year"], group_keys=False)
          .apply(_resolve_traded).reset_index(drop=True))
    fc = a[a["Pos"].apply(_is_fc)].copy()
    rows = []
    for _, row in cohort.iterrows():
        team_fc = fc[(fc["Tm"]==row["Tm"]) & (fc["season_end_year"]==row["season_end_year"])
                     & (fc["Player"]!=row["Player"]) & fc["MP"].notna() & fc["TRB%"].notna()]
        if team_fc.empty:
            val = np.nan
        else:
            top2 = team_fc.nlargest(2,"MP")
            tot  = top2["MP"].sum()
            val  = (top2["MP"]*top2["TRB%"]).sum()/tot if tot>0 else np.nan
        rows.append({"Player":row["Player"],"season_end_year":row["season_end_year"],"frontcourt_trb":val})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 5.  ERA ADJUSTMENT
# ─────────────────────────────────────────────────────────────
def add_zscore(adv_all, cohort, stat):
    """Within-season z-score of `stat` among all guards in adv_all."""
    ref = adv_all.copy()
    ref[stat] = pd.to_numeric(ref[stat], errors="coerce")
    ref["season_end_year"] = pd.to_numeric(ref["season_end_year"], errors="coerce")
    ref = ref[ref["Pos"].apply(_is_guard) & ref[stat].notna()]
    norms = (ref.groupby("season_end_year")[stat].agg(["mean","std"])
             .reset_index().rename(columns={"mean":f"{stat}_mu","std":f"{stat}_sg"}))
    out = cohort.merge(norms, on="season_end_year", how="left")
    out[f"{stat}_z"] = (out[stat]-out[f"{stat}_mu"]) / out[f"{stat}_sg"].replace(0,np.nan)
    return out.drop(columns=[f"{stat}_mu",f"{stat}_sg"])

# ─────────────────────────────────────────────────────────────
# 6.  REGRESSION + RF
# ─────────────────────────────────────────────────────────────
def run_ols(df, target_z, target_raw):
    sub  = df[[target_z]+FEATURES+["Player","season_end_year",target_raw]].dropna(subset=[target_z]+FEATURES).copy()
    X    = sm.add_constant(sub[FEATURES].astype(float))
    y    = sub[target_z].astype(float)
    mdl  = sm.OLS(y, X).fit()
    sub["predicted_z"]  = mdl.fittedvalues.values
    sub["residual_z"]   = mdl.resid.values
    sub["residual_pct"] = sub["residual_z"].rank(pct=True)*100
    return mdl, sub

def run_rf(df, target_z):
    sub = df[[target_z]+FEATURES+["Player"]].dropna().copy()
    X   = sub[FEATURES].astype(float).values
    y   = sub[target_z].astype(float).values
    rf  = RandomForestRegressor(300, max_depth=5, min_samples_leaf=5, random_state=42)
    oof = cross_val_predict(rf, X, y, cv=5)
    sub["rf_resid_z"]   = y - oof
    sub["rf_resid_pct"] = sub["rf_resid_z"].rank(pct=True)*100
    r2  = 1 - np.sum((y-oof)**2)/np.sum((y-y.mean())**2)
    print(f"    RF OOF R² = {r2:.3f}")
    return sub

def interp_coefs(mdl, target_raw):
    print(f"\n  Coefficient interpretation — {target_raw}")
    units = {"height_inches":"per extra inch of height",
             "MP":"per extra 100 min played",
             "frontcourt_trb":"per +1% frontcourt TRB% (stronger bigs)",
             "is_sg":"SG vs PG"}
    scale = {"height_inches":1,"MP":100,"frontcourt_trb":1,"is_sg":1}
    for f in FEATURES:
        c=mdl.params.get(f,np.nan); p=mdl.pvalues.get(f,np.nan); sc=scale.get(f,1)
        sig = "**" if p<0.05 else ("*" if p<0.10 else "  ")
        print(f"  {sig} {units.get(f,f)}: {c*sc:+.3f} σ  (p={p:.3f})")
    print(f"     R² = {mdl.rsquared:.3f}  n = {int(mdl.nobs)}")

# ─────────────────────────────────────────────────────────────
# 7.  FIGURES
# ─────────────────────────────────────────────────────────────
def fig_diag(mdl, target_raw):
    resid=mdl.resid; fit=mdl.fittedvalues; X=mdl.model.exog[:,1:]
    ncols=len(FEATURES)+1; fig,axes=plt.subplots(1,ncols,figsize=(4.5*ncols,4),facecolor=BG)
    for ax in axes: _dark_ax(ax)
    axes[0].scatter(fit,resid,alpha=0.4,s=16,color="#7ec8e3")
    axes[0].axhline(0,color=RED,lw=1.2,ls="--")
    axes[0].set(xlabel="Fitted (z)",ylabel="Residual (z)",title="Resid vs Fitted")
    for i,f in enumerate(FEATURES):
        axes[i+1].scatter(X[:,i],resid,alpha=0.4,s=16,color="#7ec8e3")
        axes[i+1].axhline(0,color=RED,lw=1.2,ls="--")
        axes[i+1].set(xlabel=f,ylabel="Residual (z)",title=f"vs {f}")
    fig.suptitle(f"Diagnostics — {target_raw}",color="white",fontsize=11,y=1.02)
    plt.tight_layout()
    p=os.path.join(FIGURES_DIR,f"diag_{target_raw.replace('%','pct')}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(); return p

def fig_hist(sub, target_raw, vj_z, vj_pct):
    fig,ax=plt.subplots(figsize=(10,5),facecolor=BG); _dark_ax(ax)
    resids=sub["residual_z"].dropna()
    n,_,_=ax.hist(resids,bins=28,color="#1a3a5c",edgecolor="#2a5a8c",lw=0.7)
    ax.axvline(0,color="#555",lw=1)
    ax.set(xlabel="Residual (era-adj z): Actual − Predicted",ylabel="Count",
           title=f"Distribution of {target_raw} Residuals — Rookie Guards {SEASON_START}–{SEASON_END}  (n={len(resids)})")
    if vj_z is not None:
        ax.axvline(vj_z,color=GOLD,lw=2.2,ls="--",zorder=5)
        ax.annotate(f"VJ Edgecombe\n{vj_z:+.2f} σ\n{ordinal(vj_pct)} pctile",
                    xy=(vj_z,n.max()*0.75),
                    xytext=(vj_z+max(0.3,resids.std()*0.5),n.max()*0.88),
                    color=GOLD,fontsize=9,fontweight="bold",
                    arrowprops=dict(arrowstyle="->",color=GOLD,lw=1.4))
    ax.grid(axis="y",color=GRID,lw=0.6); plt.tight_layout()
    p=os.path.join(FIGURES_DIR,f"resid_hist_{target_raw.replace('%','pct')}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(); return p

def fig_scatter(sub, target_raw, target_z, vj_row):
    fig,ax=plt.subplots(figsize=(7,7),facecolor=BG); _dark_ax(ax)
    ax.scatter(sub["predicted_z"],sub[target_z],alpha=0.35,s=20,color=BLUE,label="Cohort")
    lo=min(sub["predicted_z"].min(),sub[target_z].min())-0.2
    hi=max(sub["predicted_z"].max(),sub[target_z].max())+0.2
    ax.plot([lo,hi],[lo,hi],color="#555",lw=1,ls="--",label="Perfect prediction")
    if vj_row is not None:
        ax.scatter(vj_row["predicted_z"],vj_row[target_z],color=GOLD,s=140,zorder=6,label="VJ")
        ax.annotate("VJ",xy=(vj_row["predicted_z"],vj_row[target_z]),
                    xytext=(vj_row["predicted_z"]+0.08,vj_row[target_z]+0.12),
                    color=GOLD,fontsize=10,fontweight="bold")
    ax.set(xlabel=f"Predicted {target_raw} (z)",ylabel=f"Actual {target_raw} (z)",
           title=f"Actual vs Predicted {target_raw}")
    ax.legend(fontsize=9,facecolor="#111",labelcolor="white")
    ax.grid(color=GRID,lw=0.5); plt.tight_layout()
    p=os.path.join(FIGURES_DIR,f"scatter_{target_raw.replace('%','pct')}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(); return p

def fig_strip(summary):
    targets=["TRB%","ORB%","DRB%"]
    fig,axes=plt.subplots(1,3,figsize=(12,2.5),facecolor=BG)
    for ax,tgt in zip(axes,targets):
        _dark_ax(ax); ax.set_xlim(0,100); ax.set_ylim(-0.5,1.5)
        ax.axvline(50,color="#555",lw=1,ls=":")
        ax.barh(0,100,height=0.4,color="#1a3a5c",left=0)
        ols_p=summary.get(tgt,{}).get("ols_pct",np.nan)
        rf_p =summary.get(tgt,{}).get("rf_pct", np.nan)
        if not np.isnan(ols_p):
            ax.scatter(ols_p,0,color=GOLD,s=130,zorder=5)
            ax.text(ols_p,0.55,f"OLS {ordinal(ols_p)}",ha="center",va="bottom",color=GOLD,fontsize=8)
        if not np.isnan(rf_p):
            ax.scatter(rf_p,1,color=RED,s=80,zorder=5,marker="D")
            ax.text(rf_p,1.4,f"RF {ordinal(rf_p)}",ha="center",va="bottom",color=RED,fontsize=8)
        ax.set_title(f"{tgt} residual",color="white",fontsize=11)
        ax.set_xlabel("Percentile rank",fontsize=9); ax.set_yticks([])
        ax.set_xticks([0,25,50,75,100])
    fig.suptitle("VJ Edgecombe — Residual Percentile vs Rookie Guard Cohort",
                 color="white",fontsize=12,y=1.08)
    plt.tight_layout()
    p=os.path.join(FIGURES_DIR,"vj_percentile_strip.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close(); return p

# ─────────────────────────────────────────────────────────────
# 8.  WRITTEN VERDICT
# ─────────────────────────────────────────────────────────────
def verdict_text(summary, n_cohort, demo_mode):
    pf = lambda d,k: ordinal(d.get(k,np.nan)) if not np.isnan(d.get(k,np.nan)) else "N/A"
    rf = lambda d,k: f"{d.get(k,np.nan):+.2f} σ" if not np.isnan(d.get(k,np.nan)) else "N/A"
    trb=summary.get("TRB%",{}); orb=summary.get("ORB%",{}); drb=summary.get("DRB%",{})
    demo_note="[DEMO MODE — synthetic data, methodology only]\n\n" if demo_mode else ""
    return f"""{demo_note}VJ EDGECOMBE — PHASE 1 VERDICT
================================
Cohort: {n_cohort} rookie guards, heights 6'1"–6'5", ≥{MIN_MINUTES} min, {SEASON_START}–{SEASON_END}
Model: OLS + RF robustness check on era-adjusted within-season z-scores

HEADLINE RESULTS:
  TRB%: actual {rf(trb,'actual_z')} | predicted {rf(trb,'pred_z')} | residual {rf(trb,'resid_z')} → {pf(trb,'ols_pct')} pctile  (RF: {pf(trb,'rf_pct')})
  ORB%: actual {rf(orb,'actual_z')} | predicted {rf(orb,'pred_z')} | residual {rf(orb,'resid_z')} → {pf(orb,'ols_pct')} pctile  (RF: {pf(orb,'rf_pct')})
  DRB%: actual {rf(drb,'actual_z')} | predicted {rf(drb,'pred_z')} | residual {rf(drb,'resid_z')} → {pf(drb,'ols_pct')} pctile  (RF: {pf(drb,'rf_pct')})

MODEL FIT:
  TRB% R² = {trb.get('r2',np.nan):.3f}  |  ORB% R² = {orb.get('r2',np.nan):.3f}  |  DRB% R² = {drb.get('r2',np.nan):.3f}
  Low R² is expected and correct — the unexplained variance is precisely
  what we're hunting.  VJ's residual lives in that unexplained part.

INTERPRETATION:
  A large positive residual (>75th pctile, consistent across OLS & RF)
  = the "knack for the ball" is a real, quantifiable signal above what
  height, minutes, frontcourt teammates, and position predict.
  ORB% > DRB% residual → edge concentrates on the offensive glass.
  Consistent with scouting descriptions of anticipation and positioning.

KNOWN CAVEATS:
  - Height only in primary model (wingspan spotty; Model B robustness check included).
  - frontcourt_trb is season-total proxy, not true on/off lineup control.
  - Era z-score handles but doesn't fully eliminate 3-pt era drift.
  - Box-score rates can't distinguish contested from uncontested boards.
  These gaps are exactly what Phase 2 (tracking) and Phase 3 (CV) fill.

WHAT THIS MEANS FOR NEXT STEPS:
  IF large positive residual → box score confirms the knack; Phase 2 explains *why*.
  IF flat/negative residual → timing-based edge likely (leverage-weighted analysis needed).
"""

# ─────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────
def run(demo_mode=False):
    print("="*60)
    print(f"Phase 1 — VJ Edgecombe Rebounding Residual Study")
    print(f"Mode: {'DEMO (synthetic)' if demo_mode else 'REAL (BBRef)'}")
    print("="*60)

    # ── LOAD DATA ──
    if demo_mode:
        print("\n[1] Generating synthetic cohort…")
        cohort  = make_synthetic_cohort()
        adv_all = cohort.copy()
        # Demo z-score: within cohort (no external reference season needed)
        for stat in ["TRB%","ORB%","DRB%"]:
            mu=cohort[stat].mean(); sg=cohort[stat].std()
            cohort[f"{stat}_z"]=(cohort[stat]-mu)/sg
        print(f"  {len(cohort)-1} synthetic players + VJ row (rate stats = NaN until real run)")

    else:
        print("\n[1] Loading BBRef advanced stats (~10-12 min first run)…")
        adv_all = load_all_advanced()
        print("\n[2] Loading player heights…")
        heights = fetch_bbref_heights()
        print("\n[3] Building cohort…")
        cohort  = build_cohort(adv_all, heights)
        print("\n[4] Frontcourt TRB control…")
        fc_df   = compute_frontcourt_trb(adv_all, cohort)
        cohort  = cohort.merge(fc_df, on=["Player","season_end_year"], how="left")
        print("\n[5] Era adjustment (within-season z-score)…")
        for stat in ["TRB%","ORB%","DRB%"]:
            cohort = add_zscore(adv_all, cohort, stat)

    # ── NUMERIC COERCE ──
    for c in ["height_inches","MP","TRB%","ORB%","DRB%","frontcourt_trb","is_sg","season_end_year"]:
        if c in cohort.columns:
            cohort[c] = pd.to_numeric(cohort[c], errors="coerce")

    # ── SAVE COHORT CSV ──
    save_cols=["Player","Pos","height_inches","season_end_year","Tm","MP",
               "TRB%","ORB%","DRB%","TRB%_z","ORB%_z","DRB%_z","frontcourt_trb","is_sg"]
    cohort[[c for c in save_cols if c in cohort.columns]].to_csv(
        os.path.join(DATA_DIR,"cohort_rebounding.csv"), index=False)
    print(f"\n  Saved: data/cohort_rebounding.csv  ({len(cohort)} rows)")

    # ── MODELS ──
    summary     = {}
    all_results = []
    fig_paths   = []
    print("\n[6] Running models…")

    for target_raw in ["TRB%","ORB%","DRB%"]:
        target_z = f"{target_raw}_z"
        print(f"\n  ─── {target_raw} ───")

        mdl, sub = run_ols(cohort, target_z, target_raw)
        print(mdl.summary())
        interp_coefs(mdl, target_raw)

        vj_m = sub["Player"].str.contains("Edgecombe", na=False, case=False)
        vj_present = vj_m.any()
        vj_z    = float(sub.loc[vj_m,"residual_z"].values[0])  if vj_present else np.nan
        vj_pct  = float(sub.loc[vj_m,"residual_pct"].values[0]) if vj_present else np.nan
        vj_pred = float(sub.loc[vj_m,"predicted_z"].values[0])  if vj_present else np.nan
        vj_act  = float(sub.loc[vj_m,target_z].values[0])       if vj_present else np.nan

        # RF robustness
        rf_sub  = run_rf(cohort, target_z)
        vj_rf_m = rf_sub["Player"].str.contains("Edgecombe", na=False, case=False)
        vj_rf_p = float(rf_sub.loc[vj_rf_m,"rf_resid_pct"].values[0]) if vj_rf_m.any() else np.nan

        summary[target_raw] = {"actual_z":vj_act,"pred_z":vj_pred,
                                "resid_z":vj_z,"ols_pct":vj_pct,
                                "rf_pct":vj_rf_p,"r2":mdl.rsquared}

        if vj_present:
            print(f"\n  ★ VJ — {target_raw}: actual {vj_act:+.3f}z | pred {vj_pred:+.3f}z "
                  f"| residual {vj_z:+.3f}z | OLS {ordinal(vj_pct)} | RF {ordinal(vj_rf_p)}")
        else:
            print(f"\n  VJ not in {target_raw} model (TRB% is NaN — fill from BBRef)")

        # Figures
        fig_paths += [
            fig_diag(mdl, target_raw),
            fig_hist(sub, target_raw, vj_z, vj_pct),
            fig_scatter(sub, target_raw, target_z,
                        sub[vj_m].iloc[0] if vj_present else None),
        ]

        # Accumulate results
        r = sub[["Player","season_end_year",target_z,"predicted_z","residual_z","residual_pct"]].copy()
        r.columns=["Player","season_end_year","actual_z","predicted_z","residual_z","residual_pct"]
        r["target"]=target_raw; all_results.append(r)

    # Percentile strip
    fig_paths.append(fig_strip(summary))
    print(f"\n  Saved: figures/vj_percentile_strip.png")

    # model_results.csv
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(RESULTS_DIR,"model_results.csv"), index=False)
    print("  Saved: results/model_results.csv")

    # Written verdict
    n_cohort = cohort[cohort["Player"]!=VJ_NAME].dropna(subset=["TRB%"]).shape[0]
    vrd = verdict_text(summary, n_cohort, demo_mode)
    print("\n" + "="*60 + "\n" + vrd)
    with open(os.path.join(RESULTS_DIR,"phase1_verdict.txt"),"w") as f: f.write(vrd)
    print("  Saved: results/phase1_verdict.txt")
    print("\nPhase 1 complete.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    run(demo_mode=args.demo)
