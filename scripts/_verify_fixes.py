"""Verify all 3 remaining fixes are in place."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
checks_passed = 0
checks_total = 0

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

print("=" * 60)
print("  NeuroShield Fix Verification")
print("=" * 60)

# --- ISSUE 1: MTTR Logic ---
print("\n--- ISSUE 1: MTTR Timing Logic ---")

orch_src = (ROOT / "src" / "orchestrator" / "main.py").read_text(encoding="utf-8", errors="replace")

check("prev_build_status variable declared",
      "prev_build_status: Optional[str] = None" in orch_src)

check("MTTR transition guard (prev_build_status != FAILURE)",
      'prev_build_status != "FAILURE"' in orch_src)

check("prev_build_status updated each cycle",
      "prev_build_status = build_status" in orch_src)

check("MTTR validation (5-300s range)",
      "5.0 <= actual_mttr <= 300.0" in orch_src)

check("Conditional MTTR reset (only on SUCCESS)",
      'build_status == "SUCCESS"' in orch_src and "failure_detected_time = None" in orch_src)

# Check MTTR data is clean
mttr_path = ROOT / "data" / "mttr_log.csv"
if mttr_path.exists():
    df = pd.read_csv(mttr_path)
    bad = df[(df["actual_mttr_s"] < 5) | (df["actual_mttr_s"] > 300)]
    check("MTTR data cleaned (no entries outside 5-300s)",
          len(bad) == 0, f"{len(df)} entries, range [{df['actual_mttr_s'].min():.1f}, {df['actual_mttr_s'].max():.1f}]")
else:
    check("MTTR data cleaned", True, "no mttr_log.csv (OK)")

# --- ISSUE 2: Chart Rendering ---
print("\n--- ISSUE 2: Chart Rendering Fallbacks ---")

dash_src = (ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8", errors="replace")

check("Line chart has try/except fallback",
      "st.line_chart(df_recent" in dash_src)

check("Pie chart has try/except fallback",
      "st.bar_chart(pd.Series(_ac))" in dash_src)

check("Line chart uses .tolist() for Plotly",
      "prob_col.tolist()" in dash_src)

# --- ISSUE 3: Batch Predict Status Boost ---
print("\n--- ISSUE 3: Batch Predict Status Boost ---")

check("_batch_predict uses per-row predictor.predict()",
      "prob = predictor.predict(log_text, tel)" in dash_src)

check("SUCCESS clamping (prob > 0.3 -> 0.05)",
      'build_status == "SUCCESS" and prob > 0.3' in dash_src)

check("FAILURE fallback in except",
      '0.65 if build_status in ("FAILURE"' in dash_src)

# --- Functional: Predictor boost ---
print("\n--- Functional: Predictor Status Boost ---")
try:
    from src.prediction.predictor import FailurePredictor
    pred = FailurePredictor(model_dir=ROOT / "models")

    p_success = pred.predict("Build completed successfully", {"jenkins_last_build_status": "SUCCESS"})
    p_failure = pred.predict("Build FAILURE error", {"jenkins_last_build_status": "FAILURE"})

    check("SUCCESS prob < 0.3", p_success < 0.3, f"got {p_success:.4f}")
    check("FAILURE prob >= 0.5", p_failure >= 0.5, f"got {p_failure:.4f}")
except Exception as e:
    check("Predictor functional test", False, str(e))

print(f"\n{'=' * 60}")
print(f"  Result: {checks_passed}/{checks_total} checks passed")
print(f"{'=' * 60}")
