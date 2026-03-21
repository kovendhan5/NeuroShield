"""
NeuroShield Webhook Event Server
Receives real-time events from Jenkins and Kubernetes
Enables sub-second failure detection
"""

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)


class WebhookEventQueue:
    """Thread-safe event queue for webhook events."""

    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def put_event(self, event_type: str, data: dict, source: str) -> None:
        """Add event to queue with metadata."""
        event = {
            "type": event_type,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        try:
            self.queue.put(event, block=False)
            logger.info(f"Event queued: {event_type} from {source}")
        except queue.Full:
            logger.warning(f"Event queue full, dropping event: {event_type}")

    def get_event(self, timeout: float = 0.1) -> Optional[dict]:
        """Get next event from queue."""
        try:
            return self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def has_events(self) -> bool:
        """Check if events are queued."""
        return not self.queue.empty()


class WebhookServer:
    """Flask-based webhook receiver for Jenkins and Kubernetes."""

    def __init__(self, port: int = 9876, event_queue: Optional[WebhookEventQueue] = None):
        self.app = Flask(__name__)
        self.port = port
        self.event_queue = event_queue or WebhookEventQueue()
        self._setup_routes()

    def _setup_routes(self):
        """Setup webhook routes."""

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

        @self.app.route("/webhook/jenkins", methods=["POST"])
        def jenkins_webhook():
            """Receive Jenkins build events."""
            data = request.get_json() or {}

            # Parse Jenkins Generic Webhook Plugin format
            build_status = data.get("build", {}).get("status", "UNKNOWN")
            build_number = data.get("build", {}).get("number", "?")
            build_url = data.get("build", {}).get("url", "")

            logger.info(f"Jenkins webhook: Build #{build_number} - {build_status}")

            self.event_queue.put_event(
                event_type="jenkins_build_complete",
                data={
                    "build_number": build_number,
                    "status": build_status,
                    "url": build_url,
                    "raw": data,
                },
                source="jenkins"
            )

            return jsonify({"received": True}), 202

        @self.app.route("/webhook/kubernetes", methods=["POST"])
        def kubernetes_webhook():
            """Receive Kubernetes events."""
            data = request.get_json() or {}

            # Parse Kubernetes event webhook format
            event_type = data.get("type", "UNKNOWN")
            obj = data.get("object", {})
            kind = obj.get("kind", "Unknown")
            name = obj.get("metadata", {}).get("name", "?")
            namespace = obj.get("metadata", {}).get("namespace", "default")
            reason = obj.get("reason", "")

            logger.info(f"Kubernetes webhook: {kind}/{name} - {event_type}")

            self.event_queue.put_event(
                event_type=f"kubernetes_{kind.lower()}_{event_type.lower()}",
                data={
                    "kind": kind,
                    "name": name,
                    "namespace": namespace,
                    "reason": reason,
                    "raw": data,
                },
                source="kubernetes"
            )

            return jsonify({"received": True}), 202

        @self.app.route("/webhook/custom", methods=["POST"])
        def custom_webhook():
            """Generic webhook endpoint for custom events."""
            data = request.get_json() or {}
            event_type = data.get("event_type", "custom")

            self.event_queue.put_event(
                event_type=event_type,
                data=data,
                source="custom"
            )

            return jsonify({"received": True}), 202

    def start(self, threaded: bool = True):
        """Start webhook server."""
        logger.info(f"Starting webhook server on port {self.port}")

        if threaded:
            thread = threading.Thread(
                target=lambda: self.app.run(
                    host="0.0.0.0",
                    port=self.port,
                    debug=False,
                    use_reloader=False
                ),
                daemon=True
            )
            thread.start()
            return thread
        else:
            self.app.run(host="0.0.0.0", port=self.port, debug=False)


# Global event queue (shared with orchestrator)
_event_queue = WebhookEventQueue()


def get_event_queue() -> WebhookEventQueue:
    """Get the global event queue."""
    return _event_queue


def start_webhook_server(port: int = 9876) -> WebhookServer:
    """Start the webhook server."""
    server = WebhookServer(port=port, event_queue=_event_queue)
    server.start(threaded=True)
    return server
