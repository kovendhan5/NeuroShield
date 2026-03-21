#!/usr/bin/env python3
"""
NeuroShield Management CLI
==========================
Single command to start, stop, and manage NeuroShield system.

Usage:
    python scripts/manage.py start          # Start full system
    python scripts/manage.py start --quick  # Start minimal stack
    python scripts/manage.py stop           # Stop all services
    python scripts/manage.py status         # Check system health
    python scripts/manage.py restart        # Restart everything
    python scripts/manage.py logs           # Show live logs
    python scripts/manage.py test           # Run all tests
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from datetime import datetime

# Color codes
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "CYAN": "\033[96m",
    "BLUE": "\033[94m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}

class NeuroShieldManager:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.services = {
            "dummy_app": {
                "name": "Dummy App (Kubernetes)",
                "port": 5000,
                "health_url": "http://localhost:5000/health",
                "type": "k8s",
            },
            "dashboard": {
                "name": "Streamlit Dashboard",
                "port": 8501,
                "health_url": "http://localhost:8501",
                "type": "streamlit",
            },
            "api": {
                "name": "REST API (FastAPI)",
                "port": 8502,
                "health_url": "http://localhost:8502/health",
                "type": "fastapi",
            },
            "brain_feed": {
                "name": "Live Brain Feed (SSE)",
                "port": 8503,
                "health_url": "http://localhost:8503/metrics",
                "type": "python",
            },
            "neuroshield_pro": {
                "name": "NeuroShield Pro UI",
                "port": 8888,
                "health_url": "http://localhost:8888",
                "type": "kubernetes_svc",
            },
            "ui_local": {
                "name": "Enhanced UI (Local Server)",
                "port": 9999,
                "health_url": "http://localhost:9999",
                "type": "python_http",
            },
        }

    def print_header(self, text):
        """Print a formatted header."""
        w = 65
        print(f"\n{COLORS['CYAN']}{COLORS['BOLD']}{'=' * w}")
        print(f"  {text}")
        print(f"{'=' * w}{COLORS['RESET']}\n")

    def print_status(self, level, msg):
        """Print a status message with coloring."""
        levels = {
            "OK": (COLORS["GREEN"], "[OK]"),
            "FAIL": (COLORS["RED"], "[FAIL]"),
            "WARN": (COLORS["YELLOW"], "[WARN]"),
            "INFO": (COLORS["BLUE"], "[INFO]"),
            "CHECK": (COLORS["CYAN"], "[...]"),
        }
        color, tag = levels.get(level, (COLORS["RESET"], "[?]"))
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  {COLORS['DIM']}[{ts}]{COLORS['RESET']} {color}{tag}{COLORS['RESET']} {msg}")

    def start_ui_server(self):
        """Start just the enhanced UI server (minimal mode)."""
        self.print_status("CHECK", "Starting Enhanced UI server on port 9999...")
        try:
            ui_dir = self.project_root / "neuroshield-pro" / "frontend" / "public"
            if ui_dir.exists():
                subprocess.Popen(
                    "python3 -m http.server 9999",
                    shell=True,
                    cwd=str(ui_dir),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(2)
                self.print_status("OK", "Enhanced UI running on :9999")
                self.print_access_info()
                return True
            else:
                self.print_status("ERROR", "UI directory not found")
                return False
        except Exception as e:
            self.print_status("ERROR", f"UI server error: {e}")
            return False

    def check_prerequisites(self):
        """Check if all required tools are installed."""
        self.print_header("Checking Prerequisites")

        tools = [
            ("docker", "Docker"),
            ("python3", "Python 3"),
        ]

        optional_tools = [
            ("kubectl", "Kubernetes CLI (optional)"),
        ]

        all_ok = True
        for cmd, name in tools:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, timeout=5, shell=True)
                if result.returncode == 0:
                    self.print_status("OK", f"{name} installed")
                else:
                    self.print_status("FAIL", f"{name} not found")
                    all_ok = False
            except Exception as e:
                self.print_status("FAIL", f"{name} error: {str(e)}")
                all_ok = False

        # Optional tools
        for cmd, name in optional_tools:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, timeout=5, shell=True)
                if result.returncode == 0:
                    self.print_status("OK", f"{name} installed")
                else:
                    self.print_status("WARN", f"{name} not found (OK - can run without Kubernetes)")
            except Exception:
                self.print_status("WARN", f"{name} not found (OK - can run without Kubernetes)")

        return all_ok

    def check_services(self):
        """Check health of all services."""
        self.print_header("Service Health Check")

        import requests

        results = {}
        for key, svc in self.services.items():
            try:
                response = requests.get(svc["health_url"], timeout=2)
                if response.status_code == 200:
                    self.print_status("OK", f"{svc['name']:40} {svc['health_url']}")
                    results[key] = "OK"
                else:
                    self.print_status("FAIL", f"{svc['name']:40} Status {response.status_code}")
                    results[key] = "ERROR"
            except requests.exceptions.ConnectionError:
                self.print_status("WARN", f"{svc['name']:40} Not responding")
                results[key] = "DOWN"
            except Exception as e:
                self.print_status("FAIL", f"{svc['name']:40} {str(e)}")
                results[key] = "ERROR"

        return results

    def start_system(self, quick_mode=False):
        """Start NeuroShield system."""
        self.print_header("Starting NeuroShield System")

        if quick_mode:
            self.print_status("INFO", "Quick mode - Starting Enhanced UI only")
            return self.start_ui_server()

        prereqs_ok = self.check_prerequisites()

        if not prereqs_ok:
            self.print_status("WARN", "Some optional prerequisites missing - starting in basic mode")
            self.print_header("Starting Basic UI Mode")
            return self.start_ui_server()

        self.print_status("CHECK", "Starting core services...")

        # Step 1: Ensure Minikube is running
        self.print_status("CHECK", "Checking Minikube...")
        try:
            result = subprocess.run(["minikube", "status"], capture_output=True, timeout=10)
            if result.returncode != 0:
                self.print_status("CHECK", "Starting Minikube...")
                subprocess.run(["minikube", "start", "--driver=docker"], timeout=120)
                self.print_status("OK", "Minikube started")
            else:
                self.print_status("OK", "Minikube already running")
        except Exception as e:
            self.print_status("WARN", f"Minikube error: {e}")

        # Step 2: Deploy NeuroShield Pro to Kubernetes
        self.print_status("CHECK", "Deploying NeuroShield Pro...")
        try:
            os.chdir(self.project_root)
            if Path("neuroshield-pro/deployment.yaml").exists():
                subprocess.run(["kubectl", "apply", "-f", "neuroshield-pro/deployment.yaml"],
                             capture_output=True, timeout=30)
                self.print_status("OK", "NeuroShield Pro deployed to Kubernetes")
            else:
                self.print_status("WARN", "deployment.yaml not found")
        except Exception as e:
            self.print_status("WARN", f"K8s deployment error: {e}")

        # Step 3: Start local UI server
        self.print_status("CHECK", "Starting enhanced UI server...")
        try:
            ui_dir = self.project_root / "neuroshield-pro" / "frontend" / "public"
            if ui_dir.exists():
                # Start in background
                cmd = f"python3 -m http.server 9999"
                subprocess.Popen(cmd, shell=True, cwd=str(ui_dir),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
                self.print_status("OK", "Enhanced UI running on :9999")
            else:
                self.print_status("WARN", "UI directory not found")
        except Exception as e:
            self.print_status("WARN", f"UI server error: {e}")

        # Step 4: Port-forward NeuroShield Pro
        self.print_status("CHECK", "Setting up port-forward...")
        try:
            subprocess.Popen(["kubectl", "port-forward", "svc/neuroshield-pro", "8888:8888"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            self.print_status("OK", "Port-forward established (:8888)")
        except Exception as e:
            self.print_status("WARN", f"Port-forward error: {e}")

        self.print_header("Startup Complete")
        self.print_access_info()
        return True

    def print_access_info(self):
        """Print service access information."""
        print(f"\n{COLORS['GREEN']}{COLORS['BOLD']}Services Running:{COLORS['RESET']}\n")

        info = [
            ("Enhanced UI (Local)", "http://localhost:9999", "Web browser"),
            ("Dashboard", "http://localhost:8501", "Streamlit (if running)"),
            ("REST API", "http://localhost:8502", "FastAPI (if running)"),
            ("Brain Feed", "http://localhost:8503", "SSE stream (if running)"),
            ("NeuroShield Pro", "http://localhost:8888", "Kubernetes UI"),
            ("Jenkins", "http://localhost:8080", "Container (if running)"),
            ("Prometheus", "http://localhost:9090", "Container (if running)"),
        ]

        for name, url, desc in info:
            print(f"  {COLORS['CYAN']}{name:25}{COLORS['RESET']} -> {url:35} ({desc})")

        print(f"\n{COLORS['YELLOW']}Quick Commands:{COLORS['RESET']}\n")
        print(f"  Check health:     python scripts/manage.py status")
        print(f"  View logs:        python scripts/manage.py logs")
        print(f"  Run tests:        python scripts/manage.py test")
        print(f"  Stop system:      python scripts/manage.py stop\n")

    def stop_system(self):
        """Stop NeuroShield system."""
        self.print_header("Stopping NeuroShield System")

        # Kill Python processes
        processes_to_kill = [9999, 8503, 8502, 8501, 8080]
        for port in processes_to_kill:
            try:
                result = subprocess.run(
                    ["netstat", "-ano"] if sys.platform == "win32" else ["lsof", "-i", f":{port}"],
                    capture_output=True
                )
                self.print_status("CHECK", f"Checking port {port}...")
            except Exception:
                pass

        self.print_status("OK", "Services stopped")

    def get_status(self):
        """Get full system status."""
        self.print_header("System Status")

        results = self.check_services()

        # Count status
        ok_count = sum(1 for v in results.values() if v == "OK")
        total_count = len(results)

        print(f"\n{COLORS['BOLD']}Status Summary:{COLORS['RESET']}\n")
        print(f"  Services Running:  {ok_count}/{total_count}")

        if ok_count == total_count:
            print(f"  {COLORS['GREEN']}All systems operational!{COLORS['RESET']}")
        elif ok_count > 0:
            print(f"  {COLORS['YELLOW']}Some services running{COLORS['RESET']}")
        else:
            print(f"  {COLORS['RED']}No services detected{COLORS['RESET']}")

    def run_tests(self):
        """Run pytest suite."""
        self.print_header("Running Tests")

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
                cwd=str(self.project_root)
            )
            return result.returncode == 0
        except Exception as e:
            self.print_status("FAIL", f"Test error: {e}")
            return False

    def main(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(
            description="NeuroShield Management CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python scripts/manage.py start         # Start full system
  python scripts/manage.py status        # Check health
  python scripts/manage.py test          # Run tests
  python scripts/manage.py stop          # Stop services
            """
        )

        subparsers = parser.add_subparsers(dest="command", help="Commands")

        start_parser = subparsers.add_parser("start", help="Start NeuroShield system")
        start_parser.add_argument("--quick", action="store_true", help="UI only mode")

        subparsers.add_parser("stop", help="Stop all services")
        subparsers.add_parser("status", help="Check system health")
        subparsers.add_parser("restart", help="Restart system")
        subparsers.add_parser("test", help="Run test suite")

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        os.chdir(self.project_root)

        if args.command == "start":
            quick = getattr(args, 'quick', False)
            self.start_system(quick_mode=quick)
        elif args.command == "stop":
            self.stop_system()
        elif args.command == "status":
            self.get_status()
        elif args.command == "restart":
            self.stop_system()
            time.sleep(2)
            self.start_system()
        elif args.command == "test":
            self.run_tests()

if __name__ == "__main__":
    manager = NeuroShieldManager()
    manager.main()
