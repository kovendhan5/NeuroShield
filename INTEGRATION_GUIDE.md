"""
NeuroShield Integration Guide
How to integrate webhooks, reliability, and interpretability into main orchestrator
"""

# Add these imports to src/orchestrator/main.py:
# from src.events import (
#     get_event_queue, trace_decision, get_decision_logger,
#     get_executor, get_safety_checker, configure_reliability, ReliabilityConfig
# )
# from src.events.webhook_server import start_webhook_server

# ============================================================================
# STEP 1: Initialize in main() function
# ============================================================================

def main():
    """Enhanced main loop with webhooks, reliability, and tracing."""

    # Start webhook server (receives events from Jenkins/Kubernetes)
    logger.info("Starting webhook event server on port 9876...")
    start_webhook_server(port=9876)
    time.sleep(1)

    # Configure reliability layer
    config = ReliabilityConfig()
    config.max_retries = 3
    config.retry_delay_s = 2.0
    config.verification_timeout_s = 30.0
    configure_reliability(config)

    # Register safety checks
    safety_checker = get_safety_checker()
    safety_checker.register_check(
        "app_health_minimum",
        lambda ctx: ctx.get("app_health_pct", 100) > 5  # App can't restart if 0%
    )
    safety_checker.register_check(
        "not_rate_limited",
        lambda ctx: ctx.get("action_attempt_count", 0) < 5  # Max 5 attempts in a row
    )

    # Register fallback actions
    executor = get_executor()
    executor.register_fallback("restart_pod", FailureRecovery.pod_restart_recovery)
    executor.register_fallback("scale_up", FailureRecovery.scale_up_recovery)
    executor.register_fallback("rollback_deploy", FailureRecovery.rollback_recovery)
    executor.register_fallback("retry_build", FailureRecovery.retry_build_recovery)

    # Get event queue and decision logger
    event_queue = get_event_queue()
    decision_logger = get_decision_logger()

    logger.info("System initialized with webhooks, reliability, and tracing")

    poll_interval = int(_env("POLL_INTERVAL", "15"))

    # =========================================================================
    # Main orchestration loop (modified)
    # =========================================================================
    while True:
        cycle_start = time.time()

        # =====================================================================
        # 1. CHECK FOR WEBHOOK EVENTS (sub-second detection)
        # =====================================================================
        webhook_events = []
        while event_queue.has_events():
            event = event_queue.get_event(timeout=0.01)
            if event:
                webhook_events.append(event)
                logger.info(f"Webhook event: {event['type']} from {event['source']}")

        # =====================================================================
        # 2. COLLECT TELEMETRY (15-second polling as fallback)
        # =====================================================================
        try:
            telemetry = collect_telemetry()  # Existing function
        except Exception as e:
            logger.error(f"Telemetry collection failed: {e}")
            telemetry = {}

        # =====================================================================
        # 3. CREATE DECISION TRACE
        # =====================================================================
        import uuid
        decision_id = f"dec-{uuid.uuid4().hex[:8]}"
        trace = trace_decision(decision_id)

        # Log telemetry stage
        trace.add_stage("data_collection", {
            "sources": list(telemetry.keys()),
            "webhook_events": len(webhook_events),
        })

        # =====================================================================
        # 4. PREDICT FAILURE
        # =====================================================================
        try:
            state_vector = build_52d_state(telemetry)
            predictor = FailurePredictor()
            failure_prob = predictor.predict(state_vector)

            trace.add_stage("failure_prediction", {
                "model": "DistilBERT + PyTorch",
                "failure_probability": float(failure_prob),
                "state_dims": len(state_vector),
            })

            logger.info(f"Failure probability: {failure_prob:.3f}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            failure_prob = 0.0

        # =====================================================================
        # 5. DECIDE ACTION (with PPO agent)
        # =====================================================================
        try:
            ppo_agent = PPO.load("models/ppo_agent")
            action_id, _states = ppo_agent.predict(state_vector, deterministic=False)
            action_name = ACTION_NAMES.get(int(action_id), "unknown")

            # Determine confidence from failure probability
            confidence = max(failure_prob, 0.5)

            trace.add_stage("decision_making", {
                "agent": "PPO (Proximal Policy Optimization)",
                "action": action_name,
                "confidence": float(confidence),
                "reasoning": {
                    "failure_prob": f"{failure_prob:.3f}",
                    "webhook_triggers": len(webhook_events),
                    "action_overrides": "applied if needed",
                }
            })

        except Exception as e:
            logger.error(f"RL agent failed: {e}")
            action_name = "escalate_to_human"
            confidence = 0.5

        # =====================================================================
        # 6. SAFETY CHECKS
        # =====================================================================
        context = {
            "action": action_name,
            "app_health_pct": telemetry.get("app_health", 100),
            "action_attempt_count": 0,
        }

        is_safe, safety_reason = get_safety_checker().validate(action_name, context)
        if not is_safe:
            logger.warning(f"Safety check failed: {safety_reason}")
            action_name = "escalate_to_human"
            confidence = max(confidence - 0.1, 0.0)

        # =====================================================================
        # 7. EXECUTE WITH RELIABILITY LAYER
        # =====================================================================
        result = None
        try:
            execution_start = time.time()

            # Define execution function
            def execute_action():
                return execute_healing_action(action_name, telemetry)

            # Define verification function
            def verify_action():
                # Check if app recovered
                health = check_app_health()
                return health > 50

            # Execute with retries and fallbacks
            executor = get_executor()
            result = executor.execute(
                action_name=action_name,
                main_fn=execute_action,
                verify_fn=verify_action
            )

            execution_duration = time.time() - execution_start

            trace.add_stage("execution", {
                "command": f"kubectl action={action_name}",
                "result": "success" if result.success else "failed",
                "duration_ms": result.duration_ms,
                "retries": result.retry_count,
                "fallback_used": result.fallback_used,
            })

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result.success = False
            result.error_message = str(e)

        # =====================================================================
        # 8. RECORD DECISION IN TRACE
        # =====================================================================
        trace.set_outcome(
            success=result.success if result else False,
            result=result.error_message if result else "No result",
            duration_ms=result.duration_ms if result else 0.0
        )

        # =====================================================================
        # 9. LOG DECISION FOR INTERPRETABILITY
        # =====================================================================
        decision_logger.log_decision(trace)
        logger.info(f"Decision trace saved: {decision_id}")

        # =====================================================================
        # 10. STANDARD LOGGING (existing)
        # =====================================================================
        # ... existing logging code ...
        # _log_mttr(...), write_active_alert(...), etc.

        # =====================================================================
        # 11. WAIT FOR NEXT CYCLE
        # =====================================================================
        elapsed = time.time() - cycle_start
        sleep_time = max(0, poll_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


# ============================================================================
# ORCHESTRATOR WEBHOOK CONFIGURATION
# ============================================================================

# Tell judges where to send webhooks:

WEBHOOK_CONFIG = """
For Jenkins Generic Webhook Plugin:
  URL: http://localhost:9876/webhook/jenkins
  Event Trigger: All build events

For Kubernetes:
  Option 1 - Manual webhook (curl):
    curl -X POST http://localhost:9876/webhook/kubernetes \\
      -H "Content-Type: application/json" \\
      -d '{"type":"ADDED","object":{"kind":"Pod",...}}'

  Option 2 - Using kubelete:
    kubectl run test-pod --image=nginx
    kubectl delete pod test-pod
    (Events sent automatically to webhook)

For Custom Integrations:
  POST http://localhost:9876/webhook/custom
  {
    "event_type": "custom_event",
    "data": {...}
  }
"""

# ============================================================================
# INTEGRATION VERIFICATION
# ============================================================================

def verify_integration():
    """Verify all systems are connected."""
    checks = []

    # Check webhook server
    try:
        resp = requests.get("http://localhost:9876/health", timeout=5)
        checks.append(("Webhook Server", resp.status_code == 200))
    except:
        checks.append(("Webhook Server", False))

    # Check decision logger
    try:
        logger = get_decision_logger()
        logger.log_decision(trace_decision("test"))
        checks.append(("Decision Logger", True))
    except:
        checks.append(("Decision Logger", False))

    # Check reliability layer
    try:
        executor = get_executor()
        checks.append(("Reliability Layer", executor is not None))
    except:
        checks.append(("Reliability Layer", False))

    # Print results
    print("\n" + "="*50)
    print("INTEGRATION VERIFICATION")
    print("="*50)
    for name, status in checks:
        emoji = "✓" if status else "✗"
        print(f"  {emoji} {name}")
    print("="*50 + "\n")

    return all(status for _, status in checks)


if __name__ == "__main__":
    if verify_integration():
        main()
    else:
        print("ERROR: Integration verification failed!")
        sys.exit(1)
