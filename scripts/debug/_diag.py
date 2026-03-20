"""NeuroShield Diagnostic Script — runs all checks."""
import json, sys, pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '.')

print("=" * 60)
print("  NEUROSHIELD DIAGNOSTIC REPORT")
print(f"  Generated: {datetime.now().isoformat()}")
print("=" * 60)

# ── 1. Telemetry CSV ──
print("\n### 1. Telemetry CSV ###")
try:
    df = pd.read_csv('data/telemetry.csv', encoding='latin-1', on_bad_lines='skip')
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    cols = ['timestamp', 'jenkins_last_build_status', 'prometheus_cpu_usage', 'prometheus_memory_usage']
    avail = [c for c in cols if c in df.columns]
    print(df.tail(10)[avail].to_string())
except Exception as e:
    print(f"ERROR: {e}")

# ── 2. Healing log ──
print("\n### 2. Healing Log (last 20) ###")
try:
    lines = Path('data/healing_log.json').read_text(encoding='utf-8').strip().split('\n')
    print(f"Total entries: {len(lines)}")
    for line in lines[-20:]:
        try:
            r = json.loads(line)
            ts = r.get('timestamp', '')[:19]
            action = r.get('action_name', r.get('action', ''))
            success = r.get('success', '')
            prob = r.get('context', {}).get('failure_prob', r.get('failure_prob', ''))
            detail = r.get('detail', '')[:60]
            print(f"  {ts}  {action:20s}  success={success}  prob={prob}  {detail}")
        except:
            pass
except Exception as e:
    print(f"ERROR: {e}")

# ── 3. MTTR log ──
print("\n### 3. MTTR Log ###")
try:
    p = Path('data/mttr_log.csv')
    print(f"Exists: {p.exists()}, Size: {p.stat().st_size if p.exists() else 0}")
    if p.exists() and p.stat().st_size > 10:
        mdf = pd.read_csv(p)
        print(f"Rows: {len(mdf)}")
        print(mdf.to_string())
    else:
        print("File empty or missing")
except Exception as e:
    print(f"ERROR: {e}")

# ── 4. Cooldown check ──
print("\n### 4. Cooldown Check ###")
try:
    lines = Path('data/healing_log.json').read_text(encoding='utf-8').strip().split('\n')
    records = []
    for line in lines[-50:]:
        try:
            records.append(json.loads(line))
        except:
            pass
    violations = 0
    for i in range(1, len(records)):
        t1 = datetime.fromisoformat(records[i-1]['timestamp'].replace('Z', '+00:00'))
        t2 = datetime.fromisoformat(records[i]['timestamp'].replace('Z', '+00:00'))
        gap = (t2 - t1).total_seconds()
        if gap < 30:
            violations += 1
            a1 = records[i-1].get('action_name', records[i-1].get('action', ''))
            a2 = records[i].get('action_name', records[i].get('action', ''))
            print(f"  VIOLATION: {gap:.1f}s gap: {a1} -> {a2}")
    if violations == 0:
        print("  No cooldown violations found")
    else:
        print(f"  {violations} cooldown violations!")
except Exception as e:
    print(f"ERROR: {e}")

# ── 5. Predictor test ──
print("\n### 5. Predictor Test ###")
try:
    from src.prediction.predictor import FailurePredictor
    pred = FailurePredictor()
    df2 = pd.read_csv('data/telemetry.csv', encoding='latin-1', on_bad_lines='skip').tail(5)
    for _, row in df2.iterrows():
        log = str(row.get('jenkins_last_build_log', ''))
        if log == 'nan':
            log = ''
        tel = {
            'jenkins_last_build_status': str(row.get('jenkins_last_build_status', 'UNKNOWN')),
            'jenkins_last_build_duration': row.get('jenkins_last_build_duration', 0),
            'jenkins_queue_length': row.get('jenkins_queue_length', 0),
            'prometheus_cpu_usage': row.get('prometheus_cpu_usage', 0),
            'prometheus_memory_usage': row.get('prometheus_memory_usage', 0),
            'prometheus_pod_count': row.get('prometheus_pod_count', 0),
            'prometheus_error_rate': row.get('prometheus_error_rate', 0),
        }
        prob = pred.predict(log, tel)
        status = tel['jenkins_last_build_status']
        print(f"  Status={status:10s}  prob={prob:.4f}  log_len={len(log)}")

    # Test with enriched failure log
    print("\n  --- With enriched failure signal ---")
    enriched = ("Build step 'Execute shell' marked build as failure\n"
                "Finished: FAILURE\nERROR: script returned exit code 1\n"
                "Tests FAILED - score too low")
    prob_e = pred.predict(enriched, {'jenkins_last_build_status': 'FAILURE',
                                      'jenkins_last_build_duration': 2000,
                                      'prometheus_cpu_usage': 0,
                                      'prometheus_memory_usage': 0,
                                      'prometheus_error_rate': 0})
    print(f"  Enriched FAILURE prob: {prob_e:.4f}")

    # Test with raw failure log from CSV
    fail_rows = pd.read_csv('data/telemetry.csv', encoding='latin-1', on_bad_lines='skip')
    fail_rows = fail_rows[fail_rows['jenkins_last_build_status'] == 'FAILURE']
    if len(fail_rows) > 0:
        real_log = str(fail_rows.iloc[-1]['jenkins_last_build_log'])
        print(f"\n  Real FAILURE log length: {len(real_log)}")
        print(f"  Real FAILURE log snippet: {real_log[:200]}")
except Exception as e:
    print(f"ERROR: {e}")

# ── 6. Healing log CSV ──
print("\n### 6. Healing Log CSV ###")
try:
    hdf = pd.read_csv('data/healing_log.csv')
    print(f"Rows: {len(hdf)}, Columns: {hdf.columns.tolist()}")
    if len(hdf) > 0:
        if 'action_name' in hdf.columns:
            print(f"Action distribution:\n{hdf['action_name'].value_counts().to_string()}")
        print(f"\nLast 5:")
        print(hdf.tail(5).to_string())
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("  END DIAGNOSTIC REPORT")
print("=" * 60)
