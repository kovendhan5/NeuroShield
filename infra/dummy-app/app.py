"""NeuroShield Dummy App — Real endpoints for demo scenarios."""

import os
import sys
import threading
import time

import psutil
from flask import Flask, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_START_TIME = time.time()
_VERSION = os.getenv("APP_VERSION", "v1")
_stress_bytes: list[bytearray] = []


def _memory_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cpu_percent() -> float:
    return psutil.cpu_percent(interval=0.1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Real health check — returns 500 when version is 'v2-broken'."""
    if _VERSION == "v2-broken":
        return jsonify({"status": "error", "version": _VERSION, "reason": "bad deployment"}), 500
    return jsonify({
        "status": "ok",
        "version": _VERSION,
        "memory_mb": round(_memory_mb(), 1),
        "cpu_percent": round(_cpu_percent(), 1),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    })


@app.get("/")
def root():
    """Backward-compatible root — same as /health."""
    return health()


@app.get("/version")
def version():
    return jsonify({"version": _VERSION})


@app.post("/crash")
def crash():
    """Actually crash the process — K8s will restart the pod."""
    sys.stderr.write("CRASH requested — exiting process\n")
    sys.stderr.flush()
    # Respond before dying so the caller gets a 200
    threading.Timer(0.5, lambda: os._exit(1)).start()
    return jsonify({"status": "crashing", "message": "Process will exit in 0.5s"}), 200


@app.get("/stress")
def stress():
    """Allocate ~200 MB for 30 seconds, then release."""
    before = _memory_mb()

    def _allocate_and_release():
        global _stress_bytes
        try:
            # Allocate 200 MB in 10 MB chunks
            for _ in range(20):
                _stress_bytes.append(bytearray(10 * 1024 * 1024))
            time.sleep(30)
        finally:
            _stress_bytes.clear()

    threading.Thread(target=_allocate_and_release, daemon=True).start()
    time.sleep(1)  # let allocation start
    after = _memory_mb()
    return jsonify({
        "status": "stress_started",
        "memory_before_mb": round(before, 1),
        "memory_after_mb": round(after, 1),
        "duration_seconds": 30,
    })


@app.post("/fail")
def fail():
    """Return 500 to simulate an application error."""
    return jsonify({"status": "error", "message": "Intentional failure"}), 500


@app.get("/metrics")
def metrics():
    """Real resource metrics for monitoring."""
    return jsonify({
        "version": _VERSION,
        "memory_mb": round(_memory_mb(), 1),
        "cpu_percent": round(_cpu_percent(), 1),
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "stress_active": len(_stress_bytes) > 0,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
