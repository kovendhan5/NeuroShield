"""Patch orchestrator to add app-health-aware failure detection."""
import pathlib

p = pathlib.Path(r"K:\Devops\NeuroShield\src\orchestrator\main.py")
content = p.read_text(encoding="utf-8")

marker = "failure_prob = 0.5 if _is_failure(build_status) else 0.0"
idx = content.index(marker)
end_of_line = content.index("\n", idx)

insert_code = """
            # --- App health-aware override ---
            # If dummy-app /health is degraded (503), boost failure_prob so
            # NeuroShield acts within its 15-30s window, before K8s liveness
            # probe triggers autonomously at ~90s.
            _app_hp = app_health.get("health_pct", 100)
            if _app_hp < 100:
                _boosted = max(failure_prob, 0.78)
                if _boosted > failure_prob:
                    logging.info(
                        "[HEALTH] App degraded (health_pct=%.0f%%) -- "
                        "boosting failure_prob %.3f -> %.3f",
                        _app_hp, failure_prob, _boosted,
                    )
                    failure_prob = _boosted
"""

new_content = content[:end_of_line + 1] + insert_code + content[end_of_line + 1:]
p.write_text(new_content, encoding="utf-8")
print("DONE - inserted health-aware override after NaN guard block")
