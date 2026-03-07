"""Debug script: check data availability for dashboard charts."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import pandas as pd

# Check telemetry CSV
csv_path = Path("data/telemetry.csv")
print("=== TELEMETRY CSV ===")
print(f"Exists: {csv_path.exists()}")
if csv_path.exists():
    df = pd.read_csv(csv_path, encoding="latin-1", on_bad_lines="skip")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if "failure_prob" in df.columns:
        print(f"failure_prob sample: {df['failure_prob'].tail(5).tolist()}")
    else:
        print("failure_prob column: NOT present (computed at runtime by predictor)")
    
    print(f"\nNaN counts for prometheus columns:")
    for col in ["prometheus_cpu_usage", "prometheus_memory_usage"]:
        if col in df.columns:
            print(f"  {col}: {df[col].isna().sum()} NaN out of {len(df)}")
    
    print(f"\nLast 5 rows cpu/mem:")
    print(df[["prometheus_cpu_usage", "prometheus_memory_usage"]].tail(5))

# Check healing log
print("\n=== HEALING LOG JSON ===")
log_path = Path("data/healing_log.json")
print(f"Exists: {log_path.exists()}")
if log_path.exists():
    print(f"Size: {log_path.stat().st_size} bytes")
    content = log_path.read_text(encoding="utf-8", errors="ignore").strip()
    lines = content.splitlines()
    print(f"Lines: {len(lines)}")
    if lines:
        # Check what fields are in the records
        for line in lines[:3]:
            try:
                rec = json.loads(line)
                print(f"  Fields: {list(rec.keys())}")
                print(f"  action_name: {rec.get('action_name', 'MISSING')}")
                print(f"  action: {rec.get('action', 'MISSING')}")
                break
            except:
                pass
        
        # Count actions
        from collections import Counter
        action_counts = Counter()
        for line in lines:
            try:
                rec = json.loads(line)
                a = rec.get("action_name") or rec.get("action") or "unknown"
                action_counts[a] += 1
            except:
                pass
        print(f"  Action distribution: {dict(action_counts)}")

# Check predictor
print("\n=== PREDICTOR CHECK ===")
try:
    from src.prediction.predictor import FailurePredictor
    pred = FailurePredictor(model_dir=Path("models"))
    print("FailurePredictor loaded OK")
    # Test one prediction
    test_tel = {
        "jenkins_last_build_status": "FAILURE",
        "jenkins_last_build_duration": 5000,
        "jenkins_queue_length": 0,
        "prometheus_cpu_usage": 40.0,
        "prometheus_memory_usage": 65.0,
        "prometheus_pod_count": 1,
        "prometheus_error_rate": 0.0,
    }
    p = pred.predict("Build FAILURE: dependency timeout", test_tel)
    print(f"Test prediction (FAILURE build): {p:.4f}")
except Exception as e:
    print(f"Predictor error: {e}")
