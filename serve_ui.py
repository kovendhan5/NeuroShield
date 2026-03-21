#!/usr/bin/env python3
"""
Local HTTP server to view the enhanced UI
Serves the vibrant NeuroShield Pro UI at http://localhost:9999
"""
import http.server
import socketserver
import os
from pathlib import Path

# Change to the frontend directory
UI_DIR = Path(__file__).parent / "neuroshield-pro" / "frontend" / "public"
os.chdir(UI_DIR)

PORT = 9999

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

print(f"[SERVER] NeuroShield Pro Enhanced UI Server")
print(f"[INFO] Serving from: {UI_DIR}")
print(f"[INFO] URL: http://localhost:{PORT}")
print(f"[INFO] Features: Vibrant neon UI, Charts, Animations, Real-time updates")
print(f"\n[FEATURES] Key Improvements:")
print(f"  * Neon color scheme (#00ff88, #00ccff, #ff006e, #ffd60a)")
print(f"  * Interactive Chart.js graphs (uptime & MTTR trends)")
print(f"  * Live activity feed with slide-in animations")
print(f"  * Floating action buttons (FAB) for quick actions")
print(f"  * Toast notification system")
print(f"  * Global search bar with glow effects")
print(f"  * Glassmorphic design with backdrop blur")
print(f"  * Responsive mobile layout (3 breakpoints)")
print(f"  * Dark/Light theme toggle")
print(f"  * Advanced animations & transitions")
print(f"\n[MODULES] Coverage:")
print(f"  * Home Dashboard with 4-stat grid + 2 charts")
print(f"  * Incident Management with create/filter")
print(f"  * SLA Analytics with forecasts")
print(f"  * Pipeline Monitoring with stage view")
print(f"\n[ACTION] Press Ctrl+C to stop server")
print(f"-" * 60)

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped")
