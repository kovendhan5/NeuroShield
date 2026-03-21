"""
NeuroShield Decision Interpretability
Builds complete audit trail of every decision
Why did the AI choose this action? What was the reasoning?
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionTrace:
    """Complete record of a single healing decision."""

    def __init__(self, decision_id: str):
        self.decision_id = decision_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.stages: List[Dict[str, Any]] = []
        self.final_action: Optional[str] = None
        self.confidence: float = 0.0
        self.outcome: Optional[str] = None
        self.execution_time_ms: float = 0.0

    def add_stage(self, stage_name: str, data: Dict[str, Any], duration_ms: float = 0.0):
        """Add a processing stage to the decision trace."""
        self.stages.append({
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "data": data,
        })

    def set_decision(self, action: str, confidence: float, reasoning: Dict[str, Any]):
        """Record the final decision."""
        self.final_action = action
        self.confidence = confidence
        self.add_stage("final_decision", {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    def set_outcome(self, success: bool, result: str, duration_ms: float):
        """Record execution outcome."""
        self.outcome = "success" if success else "failure"
        self.execution_time_ms = duration_ms
        self.add_stage("execution_outcome", {
            "success": success,
            "result": result,
            "duration_ms": duration_ms,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "action": self.final_action,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "execution_time_ms": self.execution_time_ms,
            "stages": self.stages,
        }


class DecisionLogger:
    """Persistent storage of decision traces."""

    def __init__(self, log_dir: Path = Path("data/decisions")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / "decisions.jsonl"

    def log_decision(self, trace: DecisionTrace):
        """Append decision trace to JSONL file."""
        with open(self.current_log, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

    def get_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent decisions."""
        if not self.current_log.exists():
            return []

        decisions = []
        with open(self.current_log, "r") as f:
            for line in f:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return decisions[-limit:]

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific decision by ID."""
        if not self.current_log.exists():
            return None

        with open(self.current_log, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("decision_id") == decision_id:
                        return data
                except json.JSONDecodeError:
                    pass

        return None

    def get_action_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics across all decisions."""
        decisions = self.get_decisions(limit=10000)

        stats = {
            "total_decisions": len(decisions),
            "by_action": {},
            "outcome_rates": {},
            "avg_confidence": 0.0,
            "avg_execution_time_ms": 0.0,
        }

        if not decisions:
            return stats

        total_confidence = 0.0
        total_time = 0.0
        outcome_counts = {"success": 0, "failure": 0}

        for decision in decisions:
            action = decision.get("action", "unknown")
            if action not in stats["by_action"]:
                stats["by_action"][action] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "success_rate": 0.0,
                }

            stats["by_action"][action]["count"] += 1
            total_confidence += decision.get("confidence", 0.0)
            total_time += decision.get("execution_time_ms", 0.0)

            outcome = decision.get("outcome", "unknown")
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1

        # Calculate averages
        stats["avg_confidence"] = total_confidence / len(decisions)
        stats["avg_execution_time_ms"] = total_time / len(decisions)

        # Calculate outcome rates
        if len(decisions) > 0:
            stats["outcome_rates"]["success"] = outcome_counts["success"] / len(decisions)
            stats["outcome_rates"]["failure"] = outcome_counts["failure"] / len(decisions)

        # Per-action success rates
        for action in stats["by_action"]:
            action_decisions = [d for d in decisions if d.get("action") == action]
            success_count = sum(1 for d in action_decisions if d.get("outcome") == "success")
            stats["by_action"][action]["success_rate"] = success_count / len(action_decisions) if action_decisions else 0.0
            stats["by_action"][action]["avg_confidence"] = sum(d.get("confidence", 0.0) for d in action_decisions) / len(action_decisions) if action_decisions else 0.0

        return stats


# Global logger instance
_logger = DecisionLogger()


def get_decision_logger() -> DecisionLogger:
    """Get the global decision logger."""
    return _logger


def trace_decision(decision_id: str) -> DecisionTrace:
    """Create a new decision trace."""
    return DecisionTrace(decision_id)
