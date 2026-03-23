"""SQLite database models for NeuroShield persistence and audit trail."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from src.config import get_config
import json


@dataclass
class PredictionRecord:
    """Prediction record for audit trail."""
    timestamp: str
    log_hash: str
    failure_probability: float
    predicted_label: str
    actual_label: Optional[str] = None
    model_version: str = "5.0"


@dataclass
class ActionRecord:
    """Action execution record."""
    timestamp: str
    action_name: str
    state_hash: str
    reason: str
    success: bool
    duration_ms: float
    metric_change: Optional[Dict[str, float]] = None


@dataclass
class MetricSnapshot:
    """System metric snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    pod_count: int
    error_rate: float
    pod_restarts: int


class Database:
    """SQLite database for NeuroShield."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database.

        Args:
            db_path: Path to database file (defaults to config)
        """
        config = get_config()
        self.db_path = Path(db_path or config.get("database", "sqlite_path", "data/neuroshield.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    log_hash TEXT NOT NULL,
                    failure_probability REAL NOT NULL,
                    predicted_label TEXT NOT NULL,
                    actual_label TEXT,
                    model_version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Actions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    duration_ms REAL NOT NULL,
                    metric_change TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    pod_count INTEGER NOT NULL,
                    error_rate REAL NOT NULL,
                    pod_restarts INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def add_prediction(self, record: PredictionRecord) -> int:
        """Add prediction record.

        Args:
            record: Prediction record

        Returns:
            Record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions
                (timestamp, log_hash, failure_probability, predicted_label, model_version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.log_hash,
                record.failure_probability,
                record.predicted_label,
                record.model_version,
            ))
            conn.commit()
            return cursor.lastrowid

    def add_action(self, record: ActionRecord) -> int:
        """Add action record.

        Args:
            record: Action record

        Returns:
            Record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metric_json = json.dumps(record.metric_change) if record.metric_change else None
            cursor.execute("""
                INSERT INTO actions
                (timestamp, action_name, state_hash, reason, success, duration_ms, metric_change)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.action_name,
                record.state_hash,
                record.reason,
                record.success,
                record.duration_ms,
                metric_json,
            ))
            conn.commit()
            return cursor.lastrowid

    def add_metrics(self, record: MetricSnapshot) -> int:
        """Add metrics snapshot.

        Args:
            record: Metric snapshot

        Returns:
            Record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics
                (timestamp, cpu_percent, memory_percent, pod_count, error_rate, pod_restarts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp,
                record.cpu_percent,
                record.memory_percent,
                record.pod_count,
                record.error_rate,
                record.pod_restarts,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_actions(self, hours: int = 24, action_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent actions.

        Args:
            hours: Hours back to retrieve
            action_name: Optional filter by action name

        Returns:
            List of action records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if action_name:
                cursor.execute("""
                    SELECT * FROM actions
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                    AND action_name = ?
                    ORDER BY created_at DESC
                """, (hours, action_name))
            else:
                cursor.execute("""
                    SELECT * FROM actions
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                    ORDER BY created_at DESC
                """, (hours,))

            return [dict(row) for row in cursor.fetchall()]

    def get_prediction_accuracy(self, hours: int = 24) -> Dict[str, Any]:
        """Calculate prediction accuracy.

        Args:
            hours: Hours back to analyze

        Returns:
            Accuracy metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get predictions with actual labels
            cursor.execute("""
                SELECT predicted_label, actual_label
                FROM predictions
                WHERE created_at >= datetime('now', '-' || ? || ' hours')
                AND actual_label IS NOT NULL
            """, (hours,))

            results = cursor.fetchall()

            if not results:
                return {
                    "total": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                }

            correct = sum(1 for pred, actual in results if pred == actual)
            tp = sum(1 for pred, actual in results if pred == "FAILURE" and actual == "FAILURE")
            tn = sum(1 for pred, actual in results if pred == "HEALTHY" and actual == "HEALTHY")
            fp = sum(1 for pred, actual in results if pred == "FAILURE" and actual == "HEALTHY")
            fn = sum(1 for pred, actual in results if pred == "HEALTHY" and actual == "FAILURE")

            return {
                "total": len(results),
                "correct": correct,
                "accuracy": correct / len(results) if results else 0.0,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            }

    def get_action_success_rate(self, action_name: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get action success rate.

        Args:
            action_name: Optional filter by action
            hours: Hours back to analyze

        Returns:
            Success metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if action_name:
                cursor.execute("""
                    SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                    FROM actions
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                    AND action_name = ?
                """, (hours, action_name))
            else:
                cursor.execute("""
                    SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful
                    FROM actions
                    WHERE created_at >= datetime('now', '-' || ? || ' hours')
                """, (hours,))

            result = cursor.fetchone()
            total = result[0]
            successful = result[1] or 0

            return {
                "total_actions": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0.0,
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics.

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM actions")
            total_actions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM metrics")
            total_metrics = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM actions WHERE success = 1
            """)
            successful_actions = cursor.fetchone()[0]

            return {
                "total_predictions": total_predictions,
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "total_metrics": total_metrics,
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_database(db_path: Optional[str] = None) -> Database:
    """Initialize database with custom path."""
    global _db
    _db = Database(db_path)
    return _db
