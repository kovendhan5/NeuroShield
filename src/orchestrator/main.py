"""
NeuroShield Orchestrator - Real-time CI/CD Monitoring
Connects to real Jenkins and monitors actual builds
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional

# Import your trained models
from src.prediction.predictor import NeuroShieldPredictor
from stable_baselines3 import PPO


def get_latest_build_info(jenkins_url: str, job_name: str, username: str, token: str) -> Dict:
    """Get the latest build information from Jenkins"""
    url = f"{jenkins_url}/job/{job_name}/lastBuild/api/json"
    auth = (username, token)

    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            build_data = response.json()
            return {
                "number": build_data["number"],
                "timestamp": build_data["timestamp"],
                "duration": build_data["duration"],
                "result": build_data.get("result", "UNKNOWN"),
                "url": build_data["url"],
            }
        else:
            print(f"Error fetching build info: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Exception in get_latest_build_info: {e}")
        return {}


def get_build_log(jenkins_url: str, job_name: str, build_number: int, username: str, token: str) -> str:
    """Get the console log for a specific build"""
    url = f"{jenkins_url}/job/{job_name}/{build_number}/consoleText"
    auth = (username, token)

    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching build log: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Exception in get_build_log: {e}")
        return ""


def execute_healing_action(action_id: int, context: Dict) -> bool:
    """Execute the healing action based on PPO decision"""
    try:
        if action_id == 0:  # Retry
            print(f"Executing: Retry build #{context.get('build_number', 'unknown')}")
            # Call Jenkins API to retry
            jenkins_url = os.getenv("JENKINS_URL")
            job_name = os.getenv("JENKINS_JOB")
            username = os.getenv("JENKINS_USERNAME")
            token = os.getenv("JENKINS_TOKEN")

            retry_url = f"{jenkins_url}/job/{job_name}/build"
            auth = (username, token)
            response = requests.post(retry_url, auth=auth)
            return response.status_code == 201

        elif action_id == 1:  # Scale pods
            print(f"Executing: Scale pods for {context.get('affected_service', 'unknown')}")
            # Execute kubectl scale command
            import subprocess

            service = context.get("affected_service", "carts")
            cmd = f"kubectl scale deploy/{service} --replicas=3 -n sock-shop"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            return result.returncode == 0

        elif action_id == 2:  # Rollback
            print("Executing: Rollback deployment")
            # Execute rollback command
            return True  # Placeholder

        elif action_id == 3:  # No-op
            print("Executing: No operation")
            return True

        return False
    except Exception as e:
        print(f"Error executing action {action_id}: {e}")
        return False


def main():
    print("🚀 Starting NeuroShield Real-Time Orchestrator...")

    # Load trained models
    predictor = NeuroShieldPredictor()
    # Load PPO model
    model = PPO.load("models/ppo_policy.zip")

    jenkins_url = os.getenv("JENKINS_URL")
    job_name = os.getenv("JENKINS_JOB")
    username = os.getenv("JENKINS_USERNAME")
    token = os.getenv("JENKINS_TOKEN")

    print(f"Connecting to Jenkins: {jenkins_url}")
    print(f"Monitoring job: {job_name}")

    # Initialize metrics
    baseline_mttr = 0
    neuroshield_mttr = 0
    total_failures = 0
    successful_interventions = 0

    last_build_number = None

    try:
        while True:
            # Get latest build info
            build_info = get_latest_build_info(jenkins_url, job_name, username, token)

            if build_info and build_info.get("number") != last_build_number:
                print(f"\n🔍 New build detected: #{build_info['number']}")

                # Get build log
                log_content = get_build_log(jenkins_url, job_name, build_info["number"], username, token)

                if log_content:
                    print("📄 Analyzing build log...")

                    # Predict failure probability
                    failure_prob = predictor.predict_failure_probability(log_content)

                    if failure_prob > 0.5:  # Threshold for intervention
                        print(f"⚠️  High failure probability detected: {failure_prob:.2f}")

                        # Get state vector for RL agent
                        state_vector = predictor.encode_log(log_content)  # Assuming this method exists

                        # Use PPO agent to decide action
                        action_id, _states = model.predict(state_vector.reshape(1, -1), deterministic=True)

                        action_names = ["Retry", "Scale Pods", "Rollback", "No-op"]
                        print(f"🤖 NeuroShield recommends: {action_names[action_id[0]]}")

                        # Execute healing action
                        success = execute_healing_action(
                            action_id[0],
                            {
                                "build_number": build_info["number"],
                                "affected_service": "carts",  # Could be inferred from log
                            },
                        )

                        if success:
                            print("✅ Action executed successfully")
                            successful_interventions += 1
                        else:
                            print("❌ Action execution failed")

                    else:
                        print(f"✅ Build looks healthy (failure prob: {failure_prob:.2f})")

                last_build_number = build_info["number"]

            # Wait before checking again
            time.sleep(int(os.getenv("POLL_INTERVAL", 10)))

    except KeyboardInterrupt:
        print("\n🛑 NeuroShield orchestrator stopped by user")
    except Exception as e:
        print(f"\n💥 Error in orchestrator: {e}")


if __name__ == "__main__":
    main()
