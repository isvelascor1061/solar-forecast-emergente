"""
diagnose_zero_predictions.py
-----------------------------
Investigates the horizontal band of zero predictions visible in the
BiLSTM scatter plot: cases where y_pred < 10 W/m² but y_true > 200 W/m².

Sources:
  - pred_real.csv        : time, y_true, y_pred  (test-set predictions)
  - 4launch_multfeat_sym18.npz : X_test (5332, 37, 69) + feature_vars + t_test

Feature extraction: center timestep (index 18 / 37) = target hour features.
CSI_GFS proxy : kc_0100  (GFS 01 UTC clearsky index)
Daymask proxy : zenith   (feature index 68; zenith < 90 -> daytime)
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path("_4_LSTM_modules")
PRED_CSV    = ROOT / "_runs/4launch_multfeat_sym/4launch_Multfeat_sym18_BiLSTM_attn_20260428_121958/pred_real.csv"
NPZ_PATH    = ROOT / "Prepared_data/4launch_multfeat_sym18.npz"
OUT_DIR     = ROOT / "Evaluation_after_LSTM"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CENTER      = 18          # center timestep index in the 37-step window
PRED_THR    = 10.0        # W/m²  — "near-zero" prediction threshold
TRUE_THR    = 200.0       # W/m²  — minimum true radiation to count as failure

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading pred_real.csv …")
pred = pd.read_csv(PRED_CSV, parse_dates=["time"])

print("Loading 4launch_multfeat_sym18.npz …")
npz          = np.load(NPZ_PATH, allow_pickle=True)
X_test       = npz["X_test"]          # (5332, 37, 69)
t_test       = npz["t_test"]          # (5332,) datetime64[ns]
feature_vars = npz["feature_vars"].tolist()

# sanity-check alignment
assert len(pred) == len(t_test), "pred_real.csv and t_test length mismatch"
assert np.array_equal(pred["time"].values, t_test), \
    "pred_real.csv timestamps do not match t_test"

# ── 2. Extract center-timestep features ────────────────────────────────────
X_center = X_test[:, CENTER, :]       # (5332, 69)

kc_idx     = feature_vars.index("kc_0100")
zenith_idx = feature_vars.index("zenith")

df = pred.copy()
df["kc_gfs_0100"]      = X_center[:, kc_idx]
df["zenith"]           = X_center[:, zenith_idx]
df["daymask_active"]   = df["zenith"] < 90.0
df["hour"]             = df["time"].dt.hour
df["month"]            = df["time"].dt.month

# ── 3. Identify failures ────────────────────────────────────────────────────
fail_mask   = (df["y_pred"] < PRED_THR) & (df["y_true"] > TRUE_THR)
df_fail     = df[fail_mask].copy()
df_nonfail  = df[~fail_mask].copy()

n_fail      = fail_mask.sum()
n_total     = len(df)

# ── 4. Print summary ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ZERO-PREDICTION FAILURE SUMMARY")
print("="*60)
print(f"  Total test samples       : {n_total}")
print(f"  Zero-prediction failures : {n_fail}  "
      f"({100*n_fail/n_total:.1f}% of test set)")
print(f"  Mean y_true in failures  : {df_fail['y_true'].mean():.1f} W/m²")
print(f"  Max  y_true in failures  : {df_fail['y_true'].max():.1f} W/m²")

print("\n  --- Hour distribution (failures) ---")
hour_counts = df_fail["hour"].value_counts().sort_index()
for h, c in hour_counts.items():
    bar = "#" * (c // 5)
    print(f"    {h:02d}h : {c:4d}  {bar}")

print("\n  --- Top 3 failure hours ---")
for h, c in hour_counts.nlargest(3).items():
    pct = 100 * c / n_fail
    print(f"    Hour {h:02d} -> {c} failures ({pct:.1f}% of all failures)")

print("\n  --- Monthly distribution (failures) ---")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
month_counts = df_fail["month"].value_counts().sort_index()
for m, c in month_counts.items():
    print(f"    {month_names[m-1]} : {c}")
max_m = month_counts.idxmax()
print(f"  -> Peak month: {month_names[max_m-1]} ({month_counts[max_m]} failures)")
spread = month_counts.max() - month_counts.min()
print(f"  -> Range spread: {spread} — {'uniform (not seasonal)' if spread < 20 else 'possibly seasonal'}")

print("\n  --- Daymask status in failures ---")
dm = df_fail["daymask_active"].value_counts()
print(f"    Daymask active (zenith<90°) : {dm.get(True,  0)}")
print(f"    Daymask inactive            : {dm.get(False, 0)}")

print("\n  --- kc_gfs_0100 (GFS CSI) in failures ---")
print(f"    Mean  : {df_fail['kc_gfs_0100'].mean():.4f}")
print(f"    Median: {df_fail['kc_gfs_0100'].median():.4f}")
print(f"    Std   : {df_fail['kc_gfs_0100'].std():.4f}")
print(f"    Min   : {df_fail['kc_gfs_0100'].min():.4f}")
print(f"    Max   : {df_fail['kc_gfs_0100'].max():.4f}")
print(f"    Fraction where kc_gfs_0100 == 0 : "
      f"{(df_fail['kc_gfs_0100'] == 0).mean()*100:.1f}%")
print("="*60 + "\n")

# ── 5. Plot 1 — Hour of day distribution ───────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))
hours      = sorted(df_fail["hour"].unique())
all_hours  = list(range(6, 20))
counts     = [hour_counts.get(h, 0) for h in all_hours]
colors     = ["#d62728" if c > 0 else "#cccccc" for c in counts]

ax1.bar(all_hours, counts, color=colors, edgecolor="white", width=0.7)
ax1.set_xlabel("Hour of Day (UTC)", fontsize=12)
ax1.set_ylabel("Number of Failures", fontsize=12)
ax1.set_title("Zero Prediction Failures by Hour of Day\n"
              f"(y_pred < {PRED_THR} W/m²  and  y_true > {TRUE_THR} W/m²,  "
              f"n={n_fail})", fontsize=13)
ax1.set_xticks(all_hours)
ax1.set_xticklabels([f"{h:02d}h" for h in all_hours])
for h, c in zip(all_hours, counts):
    if c > 0:
        ax1.text(h, c + 3, str(c), ha="center", va="bottom", fontsize=10)
ax1.set_ylim(0, max(counts) * 1.15)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
fig1.tight_layout()
p1 = OUT_DIR / "diagnose_01_failures_by_hour.png"
fig1.savefig(p1, dpi=150)
print(f"  Saved -> {p1}")

# ── 6. Plot 2 — Scatter y_true vs hour ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
jitter     = np.random.default_rng(42).uniform(-0.3, 0.3, n_fail)
ax2.scatter(df_fail["hour"] + jitter, df_fail["y_true"],
            alpha=0.4, s=18, color="#d62728", edgecolors="none")
ax2.set_xlabel("Hour of Day (UTC)", fontsize=12)
ax2.set_ylabel("True Radiation (W/m²)", fontsize=12)
ax2.set_title("True Radiation Value When Model Predicted Zero\n"
              f"(n={n_fail}, mean y_true = {df_fail['y_true'].mean():.0f} W/m²)",
              fontsize=13)
ax2.set_xticks(all_hours)
ax2.set_xticklabels([f"{h:02d}h" for h in all_hours])
ax2.axhline(df_fail["y_true"].mean(), color="navy", linestyle="--",
            linewidth=1.2, label=f"Mean {df_fail['y_true'].mean():.0f} W/m²")
ax2.legend(fontsize=10)
ax2.grid(linestyle="--", alpha=0.4)
fig2.tight_layout()
p2 = OUT_DIR / "diagnose_02_ytrue_vs_hour.png"
fig2.savefig(p2, dpi=150)
print(f"  Saved -> {p2}")

# ── 7. Plot 3 — kc_gfs_0100 distribution: failures vs non-failures ─────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
bins = np.linspace(
    min(df["kc_gfs_0100"].min(), 0),
    max(df["kc_gfs_0100"].max(), 2),
    60
)
ax3.hist(df_nonfail["kc_gfs_0100"], bins=bins, density=True,
         alpha=0.55, color="#1f77b4", label=f"Non-failures (n={len(df_nonfail)})")
ax3.hist(df_fail["kc_gfs_0100"], bins=bins, density=True,
         alpha=0.75, color="#d62728", label=f"Zero-pred failures (n={n_fail})")
ax3.axvline(df_fail["kc_gfs_0100"].median(), color="#d62728",
            linestyle="--", linewidth=1.4,
            label=f"Failure median = {df_fail['kc_gfs_0100'].median():.3f}")
ax3.axvline(df_nonfail["kc_gfs_0100"].median(), color="#1f77b4",
            linestyle="--", linewidth=1.4,
            label=f"Non-fail median = {df_nonfail['kc_gfs_0100'].median():.3f}")
ax3.set_xlabel("kc_gfs_0100  (GFS Clearsky Index, 01 UTC run)", fontsize=12)
ax3.set_ylabel("Density", fontsize=12)
ax3.set_title("GFS Clearsky Index Distribution: Failures vs Non-Failures\n"
              "(Was GFS predicting zero when the model failed?)", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(linestyle="--", alpha=0.4)
fig3.tight_layout()
p3 = OUT_DIR / "diagnose_03_kc_gfs_distribution.png"
fig3.savefig(p3, dpi=150)
print(f"  Saved -> {p3}")

print("\nDone.")
