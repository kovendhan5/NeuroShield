# Final Visual Test — All 3 Fixes Verification

## Pre-Test Checklist

**Terminal 1 (Orchestrator)**: Kill the old session
```powershell
# Stop the current orchestrator
Ctrl+C in the orchestrator terminal

# Restart it to load the new code
cd K:\Devops\NeuroShield
python src/orchestrator/main.py --mode live
```

**Terminal 2 (Brain Feed)**: Restart to load new code
```powershell
# Stop the current Brain Feed
Ctrl+C in the brain feed terminal

# Restart
cd K:\Devops\NeuroShield
python scripts/live_brain_feed.py
```

**Terminal 3**: Ready to run crash command

---

## Video/Screenshot Capture Setup

Open all 4 simultaneously, ready for screen recording:

1. **Browser Tab A**: http://localhost:5000 (IncidentBoard)
2. **Browser Tab B**: http://localhost:8503 (Brain Feed)
3. **Browser Tab C**: http://localhost:8501 (Streamlit Dashboard)
4. **Terminal**: Orchestrator output (visible windowed)

---

## The Crash Scenario (Live Demo)

Time the event from T+0s when you trigger the crash:

```powershell
curl.exe -X POST http://localhost:5000/crash
```

### Expected Visual Timeline

| Time | Screen | Expected | Status |
|------|--------|----------|--------|
| T+2s | Tab A (IncidentBoard) | Red "SERVICE DOWN 💥" full screen with yellow blinking text | ⬜️ |
| T+11s | Terminal | Yellow/Red text: "FAILURE DETECTED" + "restart_pod" | ⬜️ |
| T+12s | Tab B (Brain Feed) | Orange/red card appears with healing action | ⬜️ |
| T+15s | Terminal | Green text: "[PORT-FORWARD] Reconnected ✓" | ⬜️ |
| T+20s | Tab A (IncidentBoard) | Green "HEALED BY NEUROSHIELD ✅" screen | ⬜️ |
| T+23s | Tab A (IncidentBoard) | Auto-reloads to normal incident list | ⬜️ |
| T+25s | Terminal | Cycle complete summary printed | ⬜️ |
| T+65s | Tab C (Dashboard) | Healing action logged in chart | ⬜️ |

---

## Per-Screen Verification Checklist

### Screen 1: IncidentBoard (localhost:5000)

**Before crash:**
- ✓ Shows incident list
- ✓ Live badge pulsing
- ✓ Incident count cards updating

**During crash (T+2s):**
- [ ] Entire page replaced with RED background
- [ ] Large text: "SERVICE DOWN"
- [ ] Crash emoji "💥"
- [ ] Blinking text: "Waiting for NeuroShield to heal..."
- [ ] No incident list visible

**Recovery (T+20s):**
- [ ] Page turns GREEN background
- [ ] Text: "HEALED BY NEUROSHIELD ✅"
- [ ] Green checkmark

**After recovery (T+23s):**
- [ ] Page returns to normal dark theme
- [ ] Incident list re-appears
- [ ] Live badge pulsing again

### Screen 2: Brain Feed (localhost:8503)

**Before crash:**
- ✓ 3-column layout visible
- ✓ Pipeline stages visible (1-6)
- ✓ Metrics cards show stats
- ✓ Heartbeat entries in middle column

**During healing (T+12s-T+15s):**
- [ ] New entry appears in middle column (Live AI Feed)
- [ ] Entry shows orange/red color: `[HH:MM:SS] restart_pod (prob=0.XXXX) → ✅ XXXms"`
- [ ] Metrics cards still visible
- [ ] Top actions still tracked

**Visual feedback:**
- [ ] One pipeline stage lights up/pulses when action fires
- [ ] Total heals metric increments

### Screen 3: Dashboard (localhost:8501)

**Before crash:**
- ✓ Charts visible with historical data
- ✓ Metrics cards showing stats
- ✓ Healing action history table populated

**During healing:**
- [ ] Live, but no immediate visual change (OK — batches updates)

**After (T+65s):**
- [ ] New row appears in healing action table
- [ ] Shows: restart_pod, SUCCESS, timestamp
- [ ] Charts update with new data point

### Screen 4: Terminal (Orchestrator)

**Before crash:**
- ✓ Cycling output every 15 seconds
- ✓ Shows CYCLE counter incrementing

**At T+11s:**
- [ ] Large red text or color: "FAILURE DETECTED" or "CRITICAL FAILURE"
- [ ] Shows: `[PREDICTOR] Failure probability: 0.XXXX CRITICAL`
- [ ] Shows: `[RL AGENT] Action selected: restart_pod`

**At T+15s:**
- [ ] Green text: `[PORT-FORWARD] Reconnected svc/dummy-app on port 5000 ✓`
- [ ] No error: "lost connection to pod"

**Summary (T+25s):**
- [ ] Shows: `Result: SUCCESS`
- [ ] Shows: `MTTR: X.Xs (baseline XXs, XX.X% reduction)`
- [ ] Shows: `Stats: X actions taken, X successful`

---

## Troubleshooting

**IncidentBoard doesn't show crash screen:**
- Check browser console for JS errors (F12)
- Verify localhost:5000 is responding to curl
- Ensure health endpoint returns 503 during crash

**Brain Feed shows no healing event:**
- Check `data/brain_feed_events.json` exists
- Tail it: `Get-Content -Tail 5 data/brain_feed_events.json`
- Verify orchestrator is writing to it (should have new entries every cycle with healing)

**Port-forward shows "lost connection":**
- Check if old kubectl process is still running
- Run: `tasklist | findstr kubectl`
- Kill manually if stuck: `taskkill /F /IM kubectl.exe`

**Terminal shows no colored output:**
- This is OK — color is nice but not critical
- Check that output otherwise matches expected text

---

## Success Criteria

**ALL PASS**:
- [ ] IncidentBoard goes red → green → normal (3 visual states)
- [ ] Brain Feed shows healing card (proof orchestrator writing events)
- [ ] Terminal shows PORT-FORWARD reconnected message (proof port-forward fixed)
- [ ] Timeline under 60 seconds (proof system is fast)

**Any FAIL**:
- Document which screen didn't respond
- Check terminal for error messages
- Verify service is still running on expected port

---

## Final Notes

These 3 fixes make the demo:
1. **Visually Dramatic** — Red crash page is unmissable
2. **Real-Time Responsive** — Brain Feed shows live AI decisions
3. **Robust** — Port-forward auto-recovers without manual intervention

Good luck with the demo! 🚀
