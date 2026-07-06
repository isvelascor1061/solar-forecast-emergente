"""
analyze_failure_features.py
----------------------------
Diagnostic: compare raw (normalized) GFS feature values between two groups:

  Group A — Dead-zone (failures) : hour in {8,9,10}, y_true > 200 W/m²
             The model predicts exactly 0.0 W/m² for EVERY sample at these hours.
             There are no successes at hours 8-10 at all, so same-hour comparison
             is impossible.

  Group B — Sunny successes      : hour in {11,12,13,14}, y_pred > 100 W/m²,
             y_true > 200 W/m²   (model works correctly at adjacent hours)

This cross-hour comparison reveals which GFS features differ between the
"dead zone" (hours 8-10 where model always outputs 0) and the adjacent
hours where the model performs correctly.

Analyses performed:
  1. Zero / near-zero feature frequency per group (table sorted by diff)
  2. Distribution comparison (mean ± std) for top 10 discriminating features
  3. Side-by-side histograms for top 6 discriminating features

IMPORTANT: Features are normalized to [0, 1] using min-max scaling over the
training set.  A value of 0.0 in the NPZ means the feature was at the minimum
observed during training — it does NOT necessarily mean the physical value was
zero (e.g. DSWRF = 0 W/m²).  The script flags this distinction in the output.

Sources:
  - 4launch_multfeat_sym18_clim.npz    : sequences (N×37×79)
  - pred_real.csv                      : test-set predictions (time, y_true, y_pred)
  - test_indices_4launch_multfeat_clim.npy : row indices into the NPZ arrays
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT       = pathlib.Path("_4_LSTM_modules")
PREP_DIR   = ROOT / "Prepared_data"
RUNS_DIR   = ROOT / "_runs" / "4launch_multfeat_sym"

NPZ_PATH   = PREP_DIR / "4launch_multfeat_sym18_clim.npz"

# Test indices — produced when the NPZ was built
TEST_IDX_PATH = ROOT / "test_indices" / "test_indices_4launch_multfeat_clim.npy"

# Best model run for the sym18+clim79 experiment
RUN_DIR    = RUNS_DIR / "4launch_Multfeat_sym18_clim79_BiLSTM_attn_20260627_081750"
PRED_CSV   = RUN_DIR / "pred_real.csv"

# Output directory
OUT_DIR    = ROOT / "Evaluation_after_LSTM" / "failure_feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis thresholds
FAILURE_HOURS = {8, 9, 10}       # hours where model outputs 0 for every sample
SUCCESS_HOURS = {11, 12, 13, 14} # adjacent hours where model works correctly
# NOTE: all samples at hours 8-10 have y_pred == 0 in this model run,
# so "failures" = all samples at those hours with y_true > TRUE_THR.
# "Successes" are taken from hours 11-14 (cross-hour comparison).
PRED_SUCCESS_THR = 100.0   # W/m² — minimum prediction to count as a success
TRUE_THR         = 200.0   # W/m² — minimum true radiation for both groups
CENTER           = 18      # Center timestep index in the 37-step window
N_GFS_FEATURES   = 69     # First 69 features are GFS; the rest are climatology

# Zero-frequency difference threshold to flag a feature
ZERO_DIFF_FLAG = 0.20  # 20 percentage points

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 68)
print("  FAILURE FEATURE ANALYSIS — hours 8, 9, 10")
print("=" * 68)

print(f"\n[1/3] Loading NPZ: {NPZ_PATH.name} …")
npz = np.load(NPZ_PATH, allow_pickle=True)
print(f"      Keys: {list(npz.keys())}")

# Retrieve feature names
feature_vars = npz["feature_vars"].tolist()
print(f"      Total features in NPZ : {len(feature_vars)}")
print(f"      GFS features used     : {N_GFS_FEATURES} (first {N_GFS_FEATURES})")

# Retrieve the test split — prefer X_test if already stored, else use test_indices
if "X_test" in npz:
    X_test = npz["X_test"]          # (N_test, 37, n_feat)
    t_test = npz["t_test"]          # (N_test,)  datetime64[ns]
    print(f"      X_test loaded directly from NPZ: shape = {X_test.shape}")
else:
    print(f"      X_test not in NPZ — loading full X and slicing with test_indices …")
    X_all  = npz["X"]               # (N_total, 37, n_feat)
    t_all  = npz["t"]               # (N_total,)
    print(f"      Full X shape: {X_all.shape}")
    print(f"      Loading test indices: {TEST_IDX_PATH.name} …")
    test_idx = np.load(TEST_IDX_PATH)
    X_test   = X_all[test_idx]
    t_test   = t_all[test_idx]
    print(f"      X_test after slicing: {X_test.shape}")

print(f"\n[2/3] Loading predictions: {PRED_CSV.name} …")
pred = pd.read_csv(PRED_CSV, parse_dates=["time"])
print(f"      Rows in pred_real.csv : {len(pred)}")

# ── 2. Align pred_real.csv with X_test ────────────────────────────────────────
# Verify that timestamps match (order must be consistent)
if len(pred) != len(t_test):
    print(f"\n  WARNING: pred_real.csv has {len(pred)} rows but t_test has "
          f"{len(t_test)} rows — cannot assert alignment.")
    print("  Proceeding with positional alignment (same order assumed).")
else:
    # Convert t_test to pandas timestamps for comparison
    t_test_pd = pd.to_datetime(t_test)
    if not np.array_equal(pred["time"].values, t_test_pd.values):
        print("  WARNING: timestamps in pred_real.csv do not exactly match t_test.")
        print("  Proceeding with positional alignment.")
    else:
        print("      Timestamps aligned — OK.")

print(f"\n[3/3] Extracting center-timestep features (index {CENTER}) …")

# ── 3. Build working DataFrame ─────────────────────────────────────────────────
# Extract features at the center timestep (the predicted hour)
X_center = X_test[:, CENTER, :]      # (N_test, n_feat)

df = pred[["time", "y_true", "y_pred"]].copy()
df["hour"] = df["time"].dt.hour

gfs_names = feature_vars[:N_GFS_FEATURES]

# Group A — failures: hours 8-10 where y_true > TRUE_THR
# (model predicts exactly 0 for ALL samples at these hours)
fail_mask  = df["hour"].isin(FAILURE_HOURS) & (df["y_true"] > TRUE_THR)
idx_fail   = np.where(fail_mask.values)[0]
X_fail     = X_center[idx_fail, :N_GFS_FEATURES]   # (N_fail, 69)

# Group B — successes: hours 11-14 where pred > threshold AND y_true > TRUE_THR
success_mask = (df["hour"].isin(SUCCESS_HOURS)
                & (df["y_pred"] > PRED_SUCCESS_THR)
                & (df["y_true"] > TRUE_THR))
idx_success  = np.where(success_mask.values)[0]
X_success    = X_center[idx_success, :N_GFS_FEATURES]   # (N_success, 69)

n_fail    = len(X_fail)
n_success = len(X_success)

print(f"\n  Group A (failures)  : {n_fail}  "
      f"— hours {sorted(FAILURE_HOURS)}, true > {TRUE_THR} W/m²")
print(f"  Group B (successes) : {n_success}  "
      f"— hours {sorted(SUCCESS_HOURS)}, pred > {PRED_SUCCESS_THR} W/m², true > {TRUE_THR} W/m²")
print(f"\n  NOTE: ALL samples at hours 8-10 have y_pred = 0.0 in this model run.")
print(f"  Comparison is cross-hour (8-10 vs 11-14), not same-hour.")

if n_fail == 0 or n_success == 0:
    print("\n  ERROR: one or both groups are empty — check thresholds / data alignment.")
    sys.exit(1)

# ── Analysis 1 — Zero / near-zero frequency ───────────────────────────────────
print("\n" + "=" * 68)
print("  ANALYSIS 1 — Zero frequency in normalized features")
print(f"  Group A: hours {sorted(FAILURE_HOURS)} (100% zero predictions)")
print(f"  Group B: hours {sorted(SUCCESS_HOURS)} (model works correctly)")
print("  NOTE: 0.0 in normalized space = training-set minimum.")
print("        This may represent a physical zero (e.g. night-time DSWRF)")
print("        OR simply a very low value.  Use this as a discriminant,")
print("        not as proof that GFS reported zero physically.")
print("=" * 68)

pct_zero_fail    = (X_fail    == 0.0).mean(axis=0) * 100.0   # (69,)
pct_zero_success = (X_success == 0.0).mean(axis=0) * 100.0   # (69,)
diff_zero        = pct_zero_fail - pct_zero_success

# Sort descending by diff
sort_idx = np.argsort(-diff_zero)

header = f"{'Feature':<30} | {'% zeros fail':>14} | {'% zeros succ':>14} | {'diff':>8}"
print(f"\n{header}")
print("-" * len(header))

flagged_features = []
for i in sort_idx:
    name  = gfs_names[i]
    zf    = pct_zero_fail[i]
    zs    = pct_zero_success[i]
    d     = diff_zero[i]
    flag  = " *** FLAGGED" if d > ZERO_DIFF_FLAG * 100 else ""
    print(f"{name:<30} | {zf:>13.1f}% | {zs:>13.1f}% | {d:>+7.1f}%{flag}")
    if d > ZERO_DIFF_FLAG * 100:
        flagged_features.append(name)

print()
if flagged_features:
    print(f"  Flagged features (diff > {ZERO_DIFF_FLAG*100:.0f}%): {flagged_features}")
else:
    print(f"  No features exceed the {ZERO_DIFF_FLAG*100:.0f}% threshold.")

# ── Analysis 2 — Distribution comparison (top 10 by mean difference) ──────────
print("\n" + "=" * 68)
print("  ANALYSIS 2 — Distribution comparison (top 10 discriminating features)")
print("=" * 68)

mean_fail    = X_fail.mean(axis=0)     # (69,)
mean_success = X_success.mean(axis=0)
std_fail     = X_fail.std(axis=0)
std_success  = X_success.std(axis=0)
mean_diff    = np.abs(mean_fail - mean_success)   # absolute difference

top10_idx = np.argsort(-mean_diff)[:10]

header2 = (f"{'Feature':<30} | {'mean(fail)':>10} | {'std(fail)':>9} | "
           f"{'mean(succ)':>10} | {'std(succ)':>9} | {'|diff|':>7} | overlap?")
print(f"\n{header2}")
print("-" * len(header2))

for i in top10_idx:
    name    = gfs_names[i]
    mf, sf  = mean_fail[i], std_fail[i]
    ms, ss  = mean_success[i], std_success[i]
    d       = mean_diff[i]

    # Simple range overlap check: [mean-std, mean+std] for both groups
    fail_lo,    fail_hi    = mf - sf, mf + sf
    success_lo, success_hi = ms - ss, ms + ss
    no_overlap = (fail_hi < success_lo) or (success_hi < fail_lo)
    overlap_str = "BARELY OVERLAPS" if no_overlap else "overlaps"

    print(f"{name:<30} | {mf:>10.4f} | {sf:>9.4f} | "
          f"{ms:>10.4f} | {ss:>9.4f} | {d:>7.4f} | {overlap_str}")

print(f"\n  NOTE: ranges are mean ± 1 std.  'BARELY OVERLAPS' means the ±1-std")
print(f"        intervals of failures and successes do not cross.")

# ── Analysis 3 — Histograms for top 6 discriminating features ─────────────────
print("\n" + "=" * 68)
print("  ANALYSIS 3 — Plotting histograms for top 6 features …")
print("=" * 68)

top6_idx = np.argsort(-mean_diff)[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for plot_i, feat_i in enumerate(top6_idx):
    ax   = axes[plot_i]
    name = gfs_names[feat_i]

    vals_fail    = X_fail[:, feat_i]
    vals_success = X_success[:, feat_i]

    all_vals = np.concatenate([vals_fail, vals_success])
    lo, hi   = all_vals.min(), all_vals.max()
    if hi - lo < 1e-8:
        lo, hi = lo - 0.1, hi + 0.1
    bins = np.linspace(lo, hi, 35)

    ax.hist(vals_success, bins=bins, density=True,
            alpha=0.6, color="#2ca02c",
            label=f"Successes (n={n_success})")
    ax.hist(vals_fail, bins=bins, density=True,
            alpha=0.7, color="#d62728",
            label=f"Failures (n={n_fail})")

    ax.axvline(vals_fail.mean(), color="#8b0000", linestyle="--",
               linewidth=1.2, label=f"Fail mean = {vals_fail.mean():.3f}")
    ax.axvline(vals_success.mean(), color="#006400", linestyle="--",
               linewidth=1.2, label=f"Succ mean = {vals_success.mean():.3f}")

    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized value [0, 1]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.4)

fig.suptitle(
    f"Feature distributions: dead-zone hours {sorted(FAILURE_HOURS)} vs sunny hours {sorted(SUCCESS_HOURS)}\n"
    f"Group A (red): hours 8-10, true > {TRUE_THR} W/m² — model outputs 0 for ALL samples\n"
    f"Group B (green): hours 11-14, pred > {PRED_SUCCESS_THR} W/m², true > {TRUE_THR} W/m²  |  "
    f"NOTE: 0.0 = training-set minimum (not necessarily physical zero)",
    fontsize=10
)
fig.tight_layout()

out_plot = OUT_DIR / "plot_feature_distributions.png"
fig.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"\n  Saved -> {out_plot}")

print("\n" + "=" * 68)
print("  Done.")
print("=" * 68)
