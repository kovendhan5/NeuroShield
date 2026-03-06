"""Quick test: what does the real predictor return for actual CSV data?"""
import sys, pandas as pd
sys.path.insert(0, '.')
from src.prediction.predictor import FailurePredictor

p = FailurePredictor()
df = pd.read_csv('data/telemetry.csv', encoding='latin-1', on_bad_lines='skip')

print(f"Total rows: {len(df)}")
print()

# Test last 5 rows
for i in range(-5, 0):
    row = df.iloc[i]
    log = str(row.get('jenkins_last_build_log', ''))
    tel = {
        'jenkins_last_build_status': row.get('jenkins_last_build_status', 'UNKNOWN'),
        'jenkins_last_build_duration': row.get('jenkins_last_build_duration', 0),
        'jenkins_queue_length': row.get('jenkins_queue_length', 0),
        'prometheus_cpu_usage': row.get('prometheus_cpu_usage', 0),
        'prometheus_memory_usage': row.get('prometheus_memory_usage', 0),
        'prometheus_pod_count': row.get('prometheus_pod_count', 0),
        'prometheus_error_rate': row.get('prometheus_error_rate', 0),
    }
    prob = p.predict(log, tel)
    status = row.get('jenkins_last_build_status', 'UNKNOWN')
    print(f"Row {len(df)+i}: status={status}, prob={prob:.4f}, log_len={len(log)}")

# Also test a SUCCESS row if any
success_rows = df[df['jenkins_last_build_status'] == 'SUCCESS']
if len(success_rows) > 0:
    row = success_rows.iloc[-1]
    log = str(row.get('jenkins_last_build_log', ''))
    tel = {
        'jenkins_last_build_status': 'SUCCESS',
        'jenkins_last_build_duration': row.get('jenkins_last_build_duration', 0),
        'jenkins_queue_length': row.get('jenkins_queue_length', 0),
        'prometheus_cpu_usage': row.get('prometheus_cpu_usage', 0),
        'prometheus_memory_usage': row.get('prometheus_memory_usage', 0),
        'prometheus_pod_count': row.get('prometheus_pod_count', 0),
        'prometheus_error_rate': row.get('prometheus_error_rate', 0),
    }
    prob = p.predict(log, tel)
    print(f"\nSUCCESS row: prob={prob:.4f}, log_len={len(log)}")
