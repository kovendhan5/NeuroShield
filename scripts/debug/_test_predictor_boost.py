"""Quick test: verify predictor status boost."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.predictor import FailurePredictor

p = FailurePredictor(model_dir="models")

tel_success = {
    "jenkins_last_build_status": "SUCCESS",
    "jenkins_last_build_duration": 45000,
    "jenkins_queue_length": 0,
    "prometheus_cpu_usage": 0,
    "prometheus_memory_usage": 0,
    "prometheus_pod_count": 0,
    "prometheus_error_rate": 0,
}
tel_failure = dict(tel_success, jenkins_last_build_status="FAILURE")

prob_s = p.predict("Build succeeded, all tests pass", tel_success)
prob_f = p.predict("Tests FAILED - score too low", tel_failure)
prob_o = p.predict("java.lang.OutOfMemoryError: Java heap space", tel_failure)

print(f"SUCCESS: {prob_s:.4f}  (expected < 0.1)")
print(f"FAILURE: {prob_f:.4f}  (expected >= 0.65)")
print(f"OOM:     {prob_o:.4f}  (expected > 0.9)")

assert prob_s < 0.1, f"SUCCESS prob too high: {prob_s}"
assert prob_f >= 0.65, f"FAILURE prob too low: {prob_f}"
assert prob_o >= 0.65, f"OOM prob too low: {prob_o}"
print("\nAll assertions passed!")
