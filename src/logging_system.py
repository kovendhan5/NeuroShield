"""
NeuroShield Advanced Logging System
Structured JSON logging with persistence, querying, and analytics
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import threading
from queue import Queue


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    source: str
    message: str
    context: Dict[str, Any]
    exception: Optional[str] = None
    trace_id: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class StructuredLogger:
    """Production-grade structured logging."""

    def __init__(self, log_dir: Path = Path("data/logs"), name: str = "neuroshield"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "neuroshield.jsonl"
        self.name = name
        self.queue = Queue()
        self.lock = threading.Lock()
        self._start_queue_processor()

    def _start_queue_processor(self):
        """Start background thread for writing queued logs."""
        thread = threading.Thread(target=self._process_queue, daemon=True)
        thread.start()

    def _process_queue(self):
        """Process log queue."""
        while True:
            try:
                entry = self.queue.get(timeout=1)
                self._write_log(entry)
            except:
                pass

    def log(
        self,
        level: str,
        message: str,
        source: str = "orchestrator",
        context: Optional[Dict] = None,
        exception: Optional[Exception] = None,
        trace_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ):
        """Log entry with context."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.upper(),
            source=source,
            message=message,
            context=context or {},
            exception=str(exception) if exception else None,
            trace_id=trace_id,
            duration_ms=duration_ms,
        )

        # Queue for async writing
        self.queue.put(entry)

        # Also print
        self._print_log(entry)

    def _print_log(self, entry: LogEntry):
        """Print log to console."""
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[92m",  # Green
            "WARN": "\033[93m",  # Yellow
            "ERROR": "\033[91m",  # Red
            "CRITICAL": "\033[95m",  # Magenta
        }

        color = colors.get(entry.level, "")
        reset = "\033[0m"

        level_tag = f"{color}[{entry.level}]{reset}"
        source_tag = f"[{entry.source}]"
        msg = f"{entry.message}"

        if entry.context:
            msg += f" {entry.context}"

        print(f"{entry.timestamp} {level_tag} {source_tag} {msg}")

    def _write_log(self, entry: LogEntry):
        """Write log to file."""
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(entry.to_json() + "\n")

    # Convenience methods
    def debug(self, message: str, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)

    def warn(self, message: str, **kwargs):
        self.log("WARN", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self.log("CRITICAL", message, **kwargs)

    # Querying
    def get_logs(
        self,
        level: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query logs with filtering."""
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)

                    # Filter
                    if level and entry.get("level") != level.upper():
                        continue
                    if source and entry.get("source") != source:
                        continue

                    entries.append(entry)
                except:
                    pass

        # Pagination
        return entries[offset : offset + limit]

    def get_recent_logs(self, hours: int = 1, level: Optional[str] = None) -> List[Dict]:
        """Get logs from last N hours."""
        if not self.log_file.exists():
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        entries = []

        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry.get("timestamp", ""))
                    if ts > cutoff:
                        if level is None or entry.get("level") == level.upper():
                            entries.append(entry)
                except:
                    pass

        return entries

    def get_statistics(self) -> Dict[str, Any]:
        """Get log statistics."""
        if not self.log_file.exists():
            return {}

        stats = {
            "total_entries": 0,
            "by_level": {},
            "by_source": {},
            "error_rate": 0.0,
        }

        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    stats["total_entries"] += 1

                    level = entry.get("level", "UNKNOWN")
                    stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

                    source = entry.get("source", "UNKNOWN")
                    stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                except:
                    pass

        if stats["total_entries"] > 0:
            error_count = stats["by_level"].get("ERROR", 0) + stats["by_level"].get("CRITICAL", 0)
            stats["error_rate"] = error_count / stats["total_entries"]

        return stats

    def clear_old_logs(self, days: int = 30):
        """Delete logs older than N days."""
        if not self.log_file.exists():
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        entries = []

        with open(self.log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry.get("timestamp", ""))
                    if ts > cutoff:
                        entries.append(line)
                except:
                    entries.append(line)

        with open(self.log_file, "w") as f:
            f.writelines(entries)


# Global logger instance
_logger = StructuredLogger()


def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    return _logger


# Usage
if __name__ == "__main__":
    logger = get_logger()

    logger.info("System starting", source="main")
    logger.debug("Debug message", source="debug", context={"value": 42})
    logger.warn("Warning occurred", source="test")
    logger.error("Error happened", source="test", context={"error_code": 500})

    # Query
    print("\nRecent logs:")
    for log in logger.get_recent_logs(hours=1):
        print(json.dumps(log))

    # Stats
    print("\nStatistics:")
    print(json.dumps(logger.get_statistics(), indent=2))
