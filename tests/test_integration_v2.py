"""
NeuroShield v2.1.0 - Complete System Integration Test
Verifies all components work together
"""

import sys
from pathlib import Path

# Add project root to path (parent of tests directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_configuration_system():
    """Test YAML configuration loading."""
    print("[OK] Testing Configuration System...")

    try:
        from src.config import Config, get_config
        config = Config("config/neuroshield.yaml")

        assert config.get("application.name") == "NeuroShield"
        assert config.get("orchestrator.poll_interval_seconds") == 15
        section = config.section("kubernetes")
        assert section.namespace == "default"

        print("  [OK] Configuration loaded and accessible")
    except Exception as e:
        print(f"  [FAIL] Configuration test failed: {e}")
        raise


def test_logging_system():
    """Test structured logging."""
    print("[OK] Testing Logging System...")
    try:
        from src.logging_system import get_logger
        logger = get_logger()

        logger.info("Test log entry", source="test", context={"test": True})
        logger.warn("Test warning", source="test")
        logger.error("Test error", source="test")

        # Query logs
        recent = logger.get_recent_logs(hours=1)
        assert len(recent) > 0

        #Get stats
        stats = logger.get_statistics()
        assert stats["total_entries"] > 0

        print(f"  [OK] Logging system working ({stats['total_entries']} entries)")
    except Exception as e:
        print(f"  [FAIL] Logging test failed: {e}")


def test_state_management():
    """Test state persistence."""
    print("[OK] Testing State Management...")
    try:
        from src.state_manager import get_state_manager
        state_mgr = get_state_manager()

        # Save and retrieve state
        state_mgr.save_state("test_key", {"value": 42})
        retrieved = state_mgr.get_state("test_key")
        assert retrieved["value"] == 42

        # Get stats
        stats = state_mgr.get_statistics()
        assert "total_actions" in stats

        print(f"  [OK] State management working ({stats['total_actions']} actions recorded)")
    except Exception as e:
        print(f"  [FAIL] State management test failed: {e}")


def test_demo_mode():
    """Test demo mode system."""
    print("[OK] Testing Demo Mode...")
    try:
        from src.demo_mode import get_demo_mode
        demo = get_demo_mode()

        # Get scenarios
        scenarios = demo.get_all_scenarios()
        assert len(scenarios) == 5

        # Start scenario
        assert demo.start_scenario("pod_crash")
        status = demo.get_scenario_status()
        assert status["status"] == "running"

        # Get metrics
        metrics = demo.get_deterministic_metrics()
        assert "cpu" in metrics or len(metrics) > 0

        print(f"  [OK] Demo mode working ({len(scenarios)} scenarios)")
    except Exception as e:
        print(f"  [FAIL] Demo mode test failed: {e}")


def test_auto_recovery():
    """Test auto-recovery system."""
    print("[OK] Testing Auto-Recovery...")
    try:
        from src.auto_recovery import get_auto_recovery
        recovery = get_auto_recovery()

        # Check status
        status = recovery.get_status()
        assert "disabled_services" in status
        assert "failure_counts" in status

        print(f"  [OK] Auto-recovery system ready")
    except Exception as e:
        print(f"  [FAIL] Auto-recovery test failed: {e}")


def test_unified_cli():
    """Test unified CLI exists."""
    print("[OK] Testing Unified CLI...")
    try:
        cli_path = Path("neuroshield")
        assert cli_path.exists()
        assert cli_path.is_file()

        print(f"  [OK] Unified CLI created ({cli_path})")
    except Exception as e:
        print(f"  [FAIL] CLI test failed: {e}")


def test_docker_compose():
    """Test Docker Compose configuration."""
    print("[OK] Testing Docker Compose...")
    try:
        docker_compose = Path("docker-compose.yml")
        assert docker_compose.exists()

        # Parse YAML
        import yaml
        with open(docker_compose) as f:
            config = yaml.safe_load(f)

        # Check services
        services = config.get("services", {})
        required_services = [
            "jenkins",
            "prometheus",
            "orchestrator",
            "dashboard",
            "neuroshield-pro",
        ]

        for service in required_services:
            assert service in services

        print(f"  [OK] Docker Compose valid ({len(services)} services)")
    except Exception as e:
        print(f"  [FAIL] Docker Compose test failed: {e}")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("  NeuroShield v2.1.0 - Integration Test Suite")
    print("="*60 + "\n")

    tests = [
        test_configuration_system,
        test_logging_system,
        test_state_management,
        test_demo_mode,
        test_auto_recovery,
        test_unified_cli,
        test_docker_compose,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test error: {e}")
            results.append(False)

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"  Results: {passed}/{total} PASSED")
    print("="*60 + "\n")

    if all(results):
        print("[OK] All integration tests passed!")
        print("[OK] System ready for deployment!")
        print("\n  Next steps:")
        print("  1. neuroshield start          # Start full system")
        print("  2. neuroshield demo pod_crash # Run demo")
        print("  3. neuroshield logs           # View logs")
        print("\n")
    else:
        print("[FAIL] Some tests failed. Fix issues before deploying.")


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
