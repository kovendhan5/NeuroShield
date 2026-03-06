"""Quick validation of retrained failure predictor."""
from src.prediction.predictor import FailurePredictor

p = FailurePredictor()

# Real Jenkins FAILURE log
prob1 = p.predict(
    "ERROR: script returned exit code 1\nBuild step Execute shell marked build as failure\nFinished: FAILURE",
    {
        "jenkins_last_build_status": "FAILURE",
        "jenkins_last_build_duration": 413000,
        "jenkins_queue_length": 5,
        "prometheus_cpu_usage": 85,
        "prometheus_memory_usage": 90,
        "prometheus_pod_count": 20,
        "prometheus_error_rate": 0.15,
    },
)

# Real SUCCESS log
prob2 = p.predict(
    "Finished: SUCCESS",
    {
        "jenkins_last_build_status": "SUCCESS",
        "jenkins_last_build_duration": 50000,
        "jenkins_queue_length": 1,
        "prometheus_cpu_usage": 25,
        "prometheus_memory_usage": 30,
        "prometheus_pod_count": 5,
        "prometheus_error_rate": 0.01,
    },
)

print(f"FAILURE log prob: {prob1:.4f}  (expect > 0.5)")
print(f"SUCCESS log prob: {prob2:.4f}  (expect < 0.5)")
print("PASS" if prob1 > 0.5 and prob2 < 0.5 else "FAIL")
