#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_run_window_comparison.py
=========================
Orchestrates BiLSTM training for sym12 and sym18 windows.
For each window:
  1. Patches SEQ_NPZ_FILE (and SEQ_MODE label) in config.py
  2. Calls LSTM_trainer.py via subprocess
  3. Restores config.py to sym24 defaults when all runs finish
"""

import os
import sys
import subprocess

BASE    = r"C:\Users\isabe\Projects\codigors\carpetasdetrabajo"
CFG     = os.path.join(BASE, "config.py")
TRAINER = os.path.join(BASE, "_4_LSTM_modules", "Main_execution_files", "LSTM_trainer.py")

RUNS = [
    # (npz_filename,                   mode_label)
    ("4launch_multfeat_sym12.npz",  "sym12"),
    ("4launch_multfeat_sym18.npz",  "sym18"),
]

# Original values to restore at the end
ORIG_NPZ  = "4launch_multfeat_sym24.npz"
ORIG_MODE = "symmetric"


def patch_config(npz_name: str, mode_label: str) -> None:
    with open(CFG, encoding="utf-8") as f:
        txt = f.read()

    # Replace the NPZ filename (only the filename, keeps f-string prefix intact)
    for candidate in [
        "4launch_multfeat_sym12.npz",
        "4launch_multfeat_sym18.npz",
        "4launch_multfeat_sym24.npz",
    ]:
        txt = txt.replace(
            f'Prepared_data/{candidate}"',
            f'Prepared_data/{npz_name}"',
        )
        # Also handle the line without closing quote (won't match but harmless)

    # Replace SEQ_MODE string value
    for candidate_mode in ["sym12", "sym18", "symmetric", "causal"]:
        txt = txt.replace(
            f'SEQ_MODE     = "{candidate_mode}"',
            f'SEQ_MODE     = "{mode_label}"',
        )

    with open(CFG, "w", encoding="utf-8") as f:
        f.write(txt)

    print(f"[orchestrator] config.py patched -> NPZ={npz_name}, MODE={mode_label}",
          flush=True)


def run_trainer(label: str) -> int:
    print(f"\n{'='*60}", flush=True)
    print(f"[orchestrator] Starting training: {label}", flush=True)
    print(f"{'='*60}", flush=True)

    env = {**os.environ, "PYTHONPATH": BASE}
    result = subprocess.run(
        [sys.executable, TRAINER],
        cwd=BASE,
        env=env,
    )
    print(f"[orchestrator] {label} finished — exit code {result.returncode}",
          flush=True)
    return result.returncode


def main():
    for npz, mode in RUNS:
        patch_config(npz, mode)
        rc = run_trainer(mode)
        if rc != 0:
            print(f"[orchestrator] WARNING: trainer returned non-zero exit code {rc}",
                  flush=True)

    # Restore config.py to sym24 defaults
    patch_config(ORIG_NPZ, ORIG_MODE)
    print("\n[orchestrator] config.py restored to sym24 defaults.", flush=True)
    print("[orchestrator] All runs complete.", flush=True)


if __name__ == "__main__":
    main()
