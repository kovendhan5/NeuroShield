#!/usr/bin/env python3
"""System verification - Check all components load correctly."""

import sys
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

# Test 6: Data directories
for dirpath in ["data", "data/escalation_reports", "logs", "models"]:
    p = Path(dirpath)
    if p.exists():
        results.append(("[OK]", f"Directory {dirpath}/ exists"))
    else:
        p.mkdir(parents=True, exist_ok=True)
        results.append(("[CREATED]", f"Directory {dirpath}/ created"))

# Test 7: Action definitions
try:
    from src.orchestrator.main import ACTION_NAMES
    if len(ACTION_NAMES) == 4:
        results.append(("[OK]", f"4 healing actions defined"))
    else:
        results.append(("[WARN]", f"Expected 4 actions, found {len(ACTION_NAMES)}"))
except Exception as e:
    results.append(("[FAIL]", f"Action check: {e}"))

# Print results
print()
for status, msg in results:
    print(f"{status} {msg}")

# Count
okays = sum(1 for s, _ in results if s in ["[OK]", "[CREATED]"])
print(f"\n=== Summary: {okays}/{len(results)} checks passed ===")

if okays == len(results):
    print("\n[SUCCESS] All systems go! Project is ready.")
    sys.exit(0)
else:
    print(f"\n[ERROR] {len(results) - okays} checks failed. Fix before proceeding.")
    sys.exit(1)
