#!/usr/bin/env python3
"""Fix pytest warnings by removing return statements from test functions."""

import re
from pathlib import Path

# Read the file
test_file = Path("k:/Devops/NeuroShield/tests/test_integration_v2.py")
content = test_file.read_text()

# Replace all "return True" that aren't in the run_all_tests function
# Pattern: function ends with return True followed by except block with return False
patterns = [
    (r"        print\(f\"  \[OK\] Logging system working \([^)]+\)\")\n        return True",
     r"        print(f\"  [OK] Logging system working ({stats['total_entries']} entries)\")"),
    (r"        print\(f\"  \[OK\] State management working \([^)]+\)\")\n        return True",
     r"        print(f\"  [OK] State management working ({stats['total_actions']} actions recorded)\")"),
    (r"        print\(f\"  \[OK\] Demo mode working \([^)]+\)\")\n        return True",
     r"        print(f\"  [OK] Demo mode working ({len(scenarios)} scenarios)\")"),
    (r"        print\(f\"  \[OK\] Auto-recovery system ready\"\)\n        return True",
     r"        print(f\"  [OK] Auto-recovery system ready\")"),
    (r"        print\(f\"  \[OK\] Unified CLI created \([^)]+\)\"\)\n        return True",
     r"        print(f\"  [OK] Unified CLI created ({cli_path})\")"),
    (r"        print\(f\"  \[OK\] Docker Compose valid \([^)]+\)\")\n        return True",
     r"        print(f\"  [OK] Docker Compose valid ({len(services)} services)\")"),
]

for old_pattern, new_pattern in patterns:
    content = re.sub(old_pattern, new_pattern, content)

# Replace all remaining "return False" with "raise" in except blocks
content = re.sub(r"        except Exception as e:\n        print\(f\"  \[FAIL\] [^\"]+\"\)\n        return False",
                 r"        except Exception as e:\n        print(f\"  [FAIL] {e}\")\n        raise",
                 content)

test_file.write_text(content)
print("Fixed test file - removed all return statements from test functions")
