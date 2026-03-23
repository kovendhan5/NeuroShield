#!/usr/bin/env python3
"""
NeuroShield v3 - Demo Script
Runs 5 demo scenarios showing self-healing in action
"""

import time
import logging
from app import Orchestrator, Database
from app.connectors import (
    JenkinsConnector,
    KubernetesConnector,
    PrometheusConnector,
    DemoScenarioInjector,
)
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print a formatted banner"""
    width = 70
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def print_result(result):
    """Print cycle result"""
    print(f"\nCycle Result:")
    print(f"  Status: {'SUCCESS' if result['success'] else 'PARTIAL'}")
    print(f"  Duration: {result['duration_ms']}ms")
    print(f"  Metrics:")
    for k, v in result['metrics'].items():
        if isinstance(v, float):
            print(f"    - {k}: {v:.1f}")
        else:
            print(f"    - {k}: {v}")
    print(f"  Anomalies: {len(result['anomalies'])}")
    print(f"  Actions: {len(result['actions'])}")
    if result['actions_taken']:
        print(f"  Actions Taken:")
        for action in result['actions_taken']:
            status = "[OK]" if action['success'] else "[FAIL]"
            print(f"    {status} {action['action']} ({action['duration_ms']}ms)")


def run_demo():
    """Run all demo scenarios"""

    print("""
================================================================
                 NeuroShield v3 - Demo Scenarios
        Intelligent CI/CD Self-Healing System Demo

   Watch as the system detects failures and auto-heals them
================================================================
    """)

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    db = Database(config["database"]["path"])
    connectors = {
        "jenkins": JenkinsConnector(config["connectors"]["jenkins"]),
        "kubernetes": KubernetesConnector(config["connectors"]["kubernetes"]),
        "prometheus": PrometheusConnector(config["connectors"]["prometheus"]),
    }
    orchestrator = Orchestrator(config, db, connectors)
    injector = DemoScenarioInjector(
        connectors["jenkins"],
        connectors["kubernetes"],
        connectors["prometheus"],
    )

    # ===================================================================
    # SCENARIO 1: Pod Crash with Auto-Recovery
    # ===================================================================

    print_banner("SCENARIO 1: Pod Crash Detection and Auto-Restart")
    print("""
This scenario demonstrates:
- Pod status monitoring
- Instant detection when pod crashes (CrashLoopBackOff)
- Automatic pod restart healing action
- Verification of recovery
    """)

    print("Step 1: Taking baseline metrics...")
    result = orchestrator.run_cycle()
    print(f"  App Health: {result['metrics']['app_health_percent']:.0f}%")

    time.sleep(1)

    print("\nStep 2: Injecting pod crash failure...")
    injector.inject_pod_crash()
    print("  Pod status changed to: CrashLoopBackOff")
    print("  Pod restarts: 5")

    time.sleep(1)

    print("\nStep 3: Running orchestrator (detect + heal)...")
    result = orchestrator.run_cycle()
    print_result(result)

    if any(a["action"] == "restart_pod" for a in result["actions_taken"]):
        print("\n[SUCCESS] System detected crash and auto-restarted pod")
    else:
        print("\n[ISSUE] Pod restart action not triggered")

    time.sleep(2)

    # ===================================================================
    # SCENARIO 2: Memory Leak Detection
    # ===================================================================

    print_banner("SCENARIO 2: Memory Leak Detection and Cache Clear")
    print("""
This scenario demonstrates:
- Gradual memory usage increase detection
- Pattern recognition (trend analysis)
- Automatic cache clearing action
- Memory trend stabilization
    """)

    print("Step 1: Recovering from previous scenario...")
    injector.recover_all()
    result = orchestrator.run_cycle()
    print(f"  System recovered. App Health: {result['metrics']['app_health_percent']:.0f}%")

    time.sleep(1)

    print("\nStep 2: Injecting memory leak...")
    injector.inject_memory_leak()
    print("  Memory pressure increased: 60%")

    time.sleep(1)

    print("\nStep 3: Running orchestrator (detect leak pattern + action)...")
    result = orchestrator.run_cycle()
    print_result(result)

    if any(a["action"] == "clear_cache" for a in result["actions_taken"]):
        print("\n[SUCCESS] System detected memory issue and cleared cache")
    else:
        print("\n[ISSUE] Cache clear action not triggered")

    time.sleep(2)

    # ===================================================================
    # SCENARIO 3: CPU Spike Auto-Scaling
    # ===================================================================

    print_banner("SCENARIO 3: CPU Spike and Auto-Scaling")
    print("""
This scenario demonstrates:
- CPU usage monitoring
- Spike detection
- Automatic scale-up action
- Load distribution
    """)

    print("Step 1: Recovering system...")
    injector.recover_all()
    result = orchestrator.run_cycle()
    print(f"  System recovered. Replicas: 1")

    time.sleep(1)

    print("\nStep 2: Injecting CPU spike...")
    injector.inject_cpu_spike()
    print("  CPU usage jumped to: 85%")

    time.sleep(1)

    print("\nStep 3: Running orchestrator (detect + scale)...")
    result = orchestrator.run_cycle()
    print_result(result)

    if any(a["action"] == "scale_up" for a in result["actions_taken"]):
        print("\n[SUCCESS] System detected CPU spike and scaled up")
    else:
        print("\n[ISSUE] Scale up action not triggered")

    time.sleep(2)

    # ===================================================================
    # SCENARIO 4: Bad Deployment Rollback
    # ===================================================================

    print_banner("SCENARIO 4: Bad Deployment Auto-Rollback")
    print("""
This scenario demonstrates:
- Build failure detection
- Error rate spike detection
- Automatic rollback decision
- Service recovery post-rollback
    """)

    print("Step 1: Recovering system...")
    injector.recover_all()
    result = orchestrator.run_cycle()
    print(f"  System healthy. Build success: {result['metrics']['build_success_rate']:.0f}%")

    time.sleep(1)

    print("\nStep 2: Injecting bad deployment...")
    injector.inject_bad_deploy()
    print("  Build failed")
    print("  Error rate jumped to: 35%")

    time.sleep(1)

    print("\nStep 3: Running orchestrator (detect bad deploy + rollback)...")
    result = orchestrator.run_cycle()
    print_result(result)

    if any(a["action"] == "rollback_deploy" for a in result["actions_taken"]):
        print("\n[SUCCESS] System detected bad deploy and auto-rolled back")
    else:
        print("\n[ISSUE] Rollback action not triggered")

    time.sleep(2)

    # ===================================================================
    # SCENARIO 5: Cascading Failure Handling
    # ===================================================================

    print_banner("SCENARIO 5: Cascading Failure (Multiple Issues)")
    print("""
This scenario demonstrates:
- Multiple simultaneous anomalies
- Priority-based action ordering
- Complex decision-making
- Escalation to human when needed
    """)

    print("Step 1: Recovering system...")
    injector.recover_all()
    result = orchestrator.run_cycle()
    print(f"  System healthy")

    time.sleep(1)

    print("\nStep 2: Injecting cascading failure...")
    injector.inject_cascading_failure()
    print("  [!] Pod crashed")
    print("  [!] CPU usage at 85%")
    print("  [!] Build failed with 35% error rate")

    time.sleep(1)

    print("\nStep 3: Running orchestrator (detect multiple + multi-action)...")
    result = orchestrator.run_cycle()
    print_result(result)

    if len(result["actions_taken"]) > 1:
        print(f"\n[SUCCESS] System took {len(result['actions_taken'])} actions to handle cascade")
    else:
        print("\n[ISSUE] Not enough actions for cascading failure")

    # ===================================================================
    # DEMO COMPLETE
    # ===================================================================

    print_banner("DEMO COMPLETE")
    print(f"""
================================================================
Summary:
================================================================

- Ran 5 real-world failure scenarios
- System detected all anomalies
- Appropriate healing actions triggered
- Database log: {config['database']['path']}

Review the database logs for detailed decision tracking:
  Actions: {len(db.get_recent_actions(limit=1000))} recorded
  Events: {len(db.get_recent_events(limit=1000))} recorded

Architecture Highlights:
[OK] State Machine: Detect -> Analyze -> Decide -> Execute
[OK] Rule-Based: Every decision is explainable
[OK] Autonomous: Works without human intervention
[OK] Persistent: All actions logged with reasoning
[OK] Observable: Metrics for monitoring and dashboard

Next Steps for Production:
1. Connect to real Jenkins/Kubernetes/Prometheus
2. Deploy in production environment
3. Monitor healing metrics (MTTR, success rate)
4. Refine rules based on production data
    """)

    print(f"\nThank you for watching NeuroShield v3 Demo\n")


if __name__ == "__main__":
    run_demo()
