"""One-time patch: add MTTR reset + summary to orchestrator main loop."""
import pathlib

p = pathlib.Path("src/orchestrator/main.py")
content = p.read_text(encoding="utf-8")

# 1. After "System healthy" line, add failure_detected_time reset
old1 = 'print(f"\\n  Status: System healthy'
idx1 = content.index(old1)
eol1 = content.index("\n", idx1)
insert1 = "\n                failure_detected_time = None  # reset MTTR timer when healthy"
content = content[:eol1] + insert1 + content[eol1:]

# 2. After "Stats:" line, add MTTR avg summary
old2 = 'print(f"\\n  Stats: {total_actions} actions taken, {successful_actions} successful")'
idx2 = content.index(old2)
eol2 = content.index("\n", idx2)
insert2 = (
    "\n            mttr_avg = sum(mttr_measurements) / len(mttr_measurements) if mttr_measurements else 0"
    '\n            if mttr_measurements:'
    '\n                print(f"  Avg MTTR Reduction: {mttr_avg:.1f}% ({len(mttr_measurements)} incidents)")'
)
content = content[:eol2] + insert2 + content[eol2:]

p.write_text(content, encoding="utf-8")
print("Patched OK")
