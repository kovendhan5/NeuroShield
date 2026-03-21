"""
NeuroShield State Persistence
SQLite-based state management with recovery and analytics
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class HealingAction:
    id: str
    timestamp: str
    action_name: str
    status: str  # success, failure, in_progress
    confidence: float
    mttr_seconds: float
    failure_type: str
    context: str  # JSON


class StateManager:
    """Manages persistent state and recovery."""

    def __init__(self, db_path: Path = Path("data/neuroshield.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Healing actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS healing_actions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action_name TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                mttr_seconds REAL,
                failure_type TEXT,
                context TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL,
                labels TEXT
            )
        """)

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT,
                data TEXT
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                severity TEXT,
                title TEXT,
                description TEXT,
                status TEXT,
                resolved_at TEXT
            )
        """)

        # Recovery states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recovery_states (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT,
                data TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_healing_action(self, action: HealingAction):
        """Save healing action."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO healing_actions
            (id, timestamp, action_name, status, confidence, mttr_seconds, failure_type, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            action.id,
            action.timestamp,
            action.action_name,
            action.status,
            action.confidence,
            action.mttr_seconds,
            action.failure_type,
            action.context,
        ))

        conn.commit()
        conn.close()

    def get_healing_actions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get healing actions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM healing_actions
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        rows = cursor.fetchall()
        conn.close()

        return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]

    def save_state(self, key: str, value: Any):
        """Save system state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO system_state (key, value)
            VALUES (?, ?)
        """, (key, json.dumps(value) if not isinstance(value, str) else value))

        conn.commit()
        conn.close()

    def get_state(self, key: str) -> Optional[Any]:
        """Get system state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        if row:
            try:
                return json.loads(row[0])
            except:
                return row[0]
        return None

    def save_metric(self, name: str, value: float, labels: Optional[Dict] = None):
        """Save metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO metrics (timestamp, metric_name, value, labels)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            name,
            value,
            json.dumps(labels) if labels else None,
        ))

        conn.commit()
        conn.close()

    def get_metrics(self, name: str, hours: int = 24) -> List[Dict]:
        """Get metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM metrics
            WHERE metric_name = ?
            AND datetime(timestamp) > datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
        """, (name, hours))

        rows = cursor.fetchall()
        conn.close()

        return rows

    def create_event(self, event_type: str, source: str, data: Dict):
        """Create event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO events (timestamp, event_type, source, data)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            event_type,
            source,
            json.dumps(data),
        ))

        conn.commit()
        conn.close()

    def create_alert(self, alert_id: str, severity: str, title: str, description: str):
        """Create alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alerts (id, timestamp, severity, title, description, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            alert_id,
            datetime.utcnow().isoformat(),
            severity,
            title,
            description,
            "active",
        ))

        conn.commit()
        conn.close()

    def resolve_alert(self, alert_id: str):
        """Resolve alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE alerts SET status = 'resolved', resolved_at = ?
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), alert_id))

        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get global statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total actions
        cursor.execute("SELECT COUNT(*) FROM healing_actions")
        total_actions = cursor.fetchone()[0]

        # Successful actions
        cursor.execute("SELECT COUNT(*) FROM healing_actions WHERE status = 'success'")
        success_count = cursor.fetchone()[0]

        # Average MTTR
        cursor.execute("SELECT AVG(mttr_seconds) FROM healing_actions WHERE status = 'success'")
        avg_mttr = cursor.fetchone()[0] or 0

        # Active alerts
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'active'")
        active_alerts = cursor.fetchone()[0]

        conn.close()

        success_rate = (success_count / total_actions * 100) if total_actions > 0 else 0

        return {
            "total_actions": total_actions,
            "successful_actions": success_count,
            "success_rate_percent": success_rate,
            "average_mttr_seconds": avg_mttr,
            "active_alerts": active_alerts,
        }

    def cleanup_old_data(self, days: int = 90):
        """Delete old data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM healing_actions
            WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
        """, (days,))

        cursor.execute("""
            DELETE FROM metrics
            WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
        """, (days,))

        conn.commit()
        conn.close()


# Global instance
_state_manager = None


def get_state_manager() -> StateManager:
    """Get global state manager."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
