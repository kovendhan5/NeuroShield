#!/usr/bin/env python3
"""System verification - Check all components load correctly."""

import sys
import json
from pathlib import Path

print("=== NeuroShield v4 - System Verification ===\n")

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

results = []

# Test 1: Orchestrator
try:
    from src.orchestrator.main import determine_healing_action, ACTION_NAMES
    results.append(("[OK]", "Orchestrator module imports"))
except Exception as e:
    results.append(("[FAIL]", f"Orchestrator: {e}"))

# Test 2: Prediction
try:
    from src.prediction.predictor import FailurePredictor, build_52d_state
    results.append(("[OK]", "Prediction module imports"))
except Exception as e:
    results.append(("[FAIL]", f"Prediction: {e}"))

# Test 3: Telemetry
try:
    from src.telemetry.collector import TelemetryCollector
    results.append(("[OK]", "Telemetry collector imports"))
except Exception as e:
    results.append(("[FAIL]", f"Telemetry: {e}"))

# Test 4: RL Agent
try:
    from src.rl_agent.simulator import simulate_action, sample_state
    results.append(("[OK]", "RL Agent simulator imports"))
except Exception as e:
    results.append(("[FAIL]", f"RL Agent: {e}"))

# Test 5: Dashboard
try:
    import streamlit
    results.append(("[OK]", "Streamlit available"))
except Exception as e:
    results.append(("[WARN]", f"Streamlit: {e}"))

# Test 6: API
try:
    from src.api.main import app, prometheus_metrics
    results.append(("[OK]", "API module imports"))
except Exception as e:
    results.append(("[FAIL]", f"API: {e}"))

# Test 7: Audit Router
try:
    from src.api.routers.audit import push_audit_event, router
    results.append(("[OK]", "Audit router imports"))
except Exception as e:
    results.append(("[FAIL]", f"Audit router: {e}"))

# Test 8: Data directories
for dirpath in ["data", "data/escalation_reports", "logs", "models", "data/logs"]:
    p = Path(dirpath)
    if p.exists():
        results.append(("[OK]", f"Directory {dirpath}/ exists"))
    else:
        p.mkdir(parents=True, exist_ok=True)
        results.append(("[CREATED]", f"Directory {dirpath}/ created"))

# Test 9: Data files
data_files = [
    "data/pipeline_runtime.json",
    "data/healing_log.json",
]
for filepath in data_files:
    p = Path(filepath)
    if p.exists():
        results.append(("[OK]", f"Data file {filepath} exists"))
    else:
        results.append(("[WARN]", f"Data file {filepath} missing"))

# Test 10: Pipeline runtime format validation
try:
    with open("data/pipeline_runtime.json") as f:
        runtime = json.load(f)
        pipelines = runtime.get("pipelines", [])
        if len(pipelines) == 4:
            results.append(("[OK]", "4 pipelines configured"))
            # Check autoheal > fails for all pipelines
            all_autoheal_ok = True
            for p in pipelines:
                if p.get("autoheal_actions", 0) <= p.get("failed_runs", 0):
                    all_autoheal_ok = False
                    results.append(("[WARN]", f"Pipeline {p['id']}: autoheal <= fails"))
            if all_autoheal_ok:
                results.append(("[OK]", "All pipelines: autoheal > fails"))
        else:
            results.append(("[WARN]", f"Expected 4 pipelines, found {len(pipelines)}"))
        
        # Check kubernetes section
        k8s = runtime.get("kubernetes", {})
        if k8s:
            results.append(("[OK]", f"Kubernetes section present (health: {k8s.get('cluster_health', 'N/A')}%)"))
        else:
            results.append(("[WARN]", "No kubernetes section in runtime"))
except Exception as e:
    results.append(("[FAIL]", f"Pipeline runtime check: {e}"))

# Test 11: Action definitions
try:
    from src.orchestrator.main import ACTION_NAMES
    if len(ACTION_NAMES) >= 4:
        results.append(("[OK]", f"{len(ACTION_NAMES)} healing actions defined"))
    else:
        results.append(("[WARN]", f"Expected >=4 actions, found {len(ACTION_NAMES)}"))
except Exception as e:
    results.append(("[FAIL]", f"Action check: {e}"))

# Test 12: Jenkinsfiles
jenkinsfiles = [
    "infra/jenkins/Jenkinsfile",
    "infra/jenkins/Jenkinsfile.ml-inference",
    "infra/jenkins/Jenkinsfile.dashboard",
    "infra/jenkins/Jenkinsfile.gitops",
]
for jf in jenkinsfiles:
    p = Path(jf)
    if p.exists():
        results.append(("[OK]", f"Jenkinsfile exists: {jf}"))
    else:
        results.append(("[WARN]", f"Missing Jenkinsfile: {jf}"))

# Test 13: Dashboard build
dashboard_dist = Path("dashboard/dist")
if dashboard_dist.exists() and list(dashboard_dist.glob("*.html")):
    results.append(("[OK]", "Dashboard built (dist/ exists)"))
else:
    results.append(("[WARN]", "Dashboard not built (run: cd dashboard && npm run build)"))

# Print results
print()
for status, msg in results:
    color = ""
    if status == "[OK]":
        color = "\033[92m"  # Green
    elif status == "[WARN]":
        color = "\033[93m"  # Yellow
    elif status == "[FAIL]":
        color = "\033[91m"  # Red
    elif status == "[CREATED]":
        color = "\033[94m"  # Blue
    reset = "\033[0m"
    print(f"{color}{status}{reset} {msg}")

# Count
okays = sum(1 for s, _ in results if s in ["[OK]", "[CREATED]"])
warns = sum(1 for s, _ in results if s == "[WARN]")
fails = sum(1 for s, _ in results if s == "[FAIL]")
print(f"\n=== Summary: {okays} OK | {warns} WARN | {fails} FAIL ===")

if fails == 0:
    print("\n[SUCCESS] All critical checks passed! Project is ready.")
    sys.exit(0)
else:
    print(f"\n[ERROR] {fails} critical checks failed. Fix before proceeding.")
    sys.exit(1)
