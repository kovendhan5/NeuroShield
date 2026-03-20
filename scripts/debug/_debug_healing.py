import json, os

# Check the actual file
path1 = "K:/Devops/NeuroShield/data/healing_log.json"
print(f"Path1 exists: {os.path.exists(path1)}")

with open(path1) as f:
    d = json.load(f)
print(f"Type: {type(d).__name__}")
if isinstance(d, list):
    print(f"Length: {len(d)}")
    if d:
        print(f"First entry keys: {list(d[0].keys()) if isinstance(d[0], dict) else d[0]}")
        print(f"First entry: {d[0]}")
elif isinstance(d, dict):
    print(f"Keys: {list(d.keys())}")
    print(f"Content preview: {str(d)[:500]}")

# Check path resolution from src/api/main.py
api_file = "K:/Devops/NeuroShield/src/api/main.py"
resolved = os.path.join(os.path.dirname(api_file), "../../data/healing_log.json")
resolved = os.path.normpath(resolved)
print(f"\nResolved path: {resolved}")
print(f"Resolved exists: {os.path.exists(resolved)}")
