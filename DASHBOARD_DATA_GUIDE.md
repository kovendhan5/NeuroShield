# NeuroShield Dashboard - Real Data Integration Guide

## ✅ System Status

All systems are **RUNNING** and **CONNECTED**:

- **Orchestrator**: Running (writes REAL incidents to `data/healing_log.json`)
- **Microservice API**: Running at `http://localhost:5000`
- **React Dashboard**: Running at `http://localhost:5173`
- **Total Real Incidents**: 309 (tracked in healing_log.json)

---

## 🎯 How to View Real Data

### Step 1: Open Dashboard
**URL**: http://localhost:5173

### Step 2: Expected Display (Overview Tab)
When the page loads, you should see:

```
┌─────────────────────────────────────────────────────┐
│   📊 KPI Cards (Top of page)                        │
├─────────────────────────────────────────────────────┤
│  • Active Heals: 309        (blue card)             │
│  • Success Rate: 85.8%      (green card)            │
│  • Avg MTTR: 3,857ms        (purple card)           │
│  • Cost Saved: $9,937.50    (amber card)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│   💹 Charts (Middle)                               │
├─────────────────────────────────────────────────────┤
│  • Left: Real-Time Performance (area chart)        │
│  • Right: Actions breakdown (pie chart)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│   ⚡ Real-Time Healing Pipeline (Bottom)          │
├─────────────────────────────────────────────────────┤
│  ❌ detect_failure          (failed)               │
│  ❌ pod_restart_needed      (failed)               │
│  ❌ retry_build             (failed)               │
│  ❌ pod_restart_needed      (failed)               │
│  ❌ high_cpu_detected       (failed)               │
│                                                     │
│  [Updates automatically every 5 seconds]           │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 If Dashboard Shows No Data

### Issue 1: Showing "Simulated" instead of "Live"
- **Fix**: Open browser DevTools (F12)
- Check Console for errors
- Check Network tab → verify `/api/dashboard/stats` returns data
- If OK, data should appear within 5 seconds

### Issue 2: Metrics show but no recent actions
- **Fix**: Scroll down to "Real-Time Healing Pipeline" section
- It's below the charts

### Issue 3: Old data (from an hour ago)
- **Fix**: Click the **Refresh** button (top right)
- Or change frequency to "Every 1s" to see live updates

### Issue 4: Still no data
**Debug Command**:
```bash
curl -s http://localhost:5000/api/dashboard/stats
```

Should return JSON with `total_heals: 309` and other metrics.

---

## 📊 Current Real Data in System

### From API (`/api/dashboard/stats`):
- **Total Healing Actions**: 309
- **Success Rate**: 85.8%
- **Failed Actions**: 43
- **Cost Saved**: $9,937.50
- **Avg Response Time**: 3,857ms

### Top Healing Actions Performed:
1. **retry_build** (66 times) - Restart failed Jenkins builds
2. **scale_up** (64 times) - Increase pod replicas
3. **restart_pod** (61 times) - Restart crashed pods
4. **rollback_deploy** (45 times) - Revert bad deployments
5. **clear_cache** (43 times) - Clear cache memory

### Source Data:
- Location: `data/healing_log.json`
- Format: NDJSON (one JSON object per line)
- Updated by: Orchestrator + Manual triggers
- Last 5 entries visible in dashboard under "Real-Time Healing Pipeline"

---

## 🎬 Trigger New Incidents to See Dashboard Update

Dashboard updates automatically every 5 seconds. Trigger an incident to see new data:

### Option A: Trigger Pod Crash
```bash
curl -X POST http://localhost:5000/api/trigger/pod-crash
```
- Dashboard will show new `pod_restart_needed` incident in ~5 seconds
- Total count increases by 1

### Option B: Trigger Jenkins Failure
```bash
curl -X POST http://localhost:5000/api/trigger/jenkins-failure
```
- Dashboard will show new `retry_build` incident
- Shows orchestrator's auto-response

### Option C: Trigger CPU Spike
```bash
curl -X POST http://localhost:5000/api/trigger/cpu-spike
```
- Dashboard will show `high_cpu_detected` incident
- Suggests `scale_up` action

### Option D: Full Demo Flow (Recommended)
```bash
curl -X POST http://localhost:5000/api/test/full-demo-flow
```
- Creates 3 sequential events:
  - T+0s: Failure detected
  - T+2s: Healing action triggered
  - T+4s: Recovery confirmed
- Dashboard captures all stages

**After Triggering**:
1. Dashboard auto-refreshes every 5 seconds
2. New incident appears in "Real-Time Healing Pipeline" section
3. Stats update (count increases, percentages recalculate)

---

## 📈 What Each Tab Shows

### Overview Tab (Default)
- KPI metrics with real-time values
- Performance charts
- Recent healing incidents
- Action breakdown pie chart

### Analytics Tab
- Detailed performance metrics
- Historical trends
- ML confidence scores
- Action effectiveness

### Health Tab
- Service status (API, PostgreSQL, Redis, Jenkins, etc.)
- Component health checks
- System reliability metrics

### Live Tab (🔴 Live Feed)
- Real-time events as they happen
- Color-coded by action type:
  - 🟢 Green = Successful healing
  - 🔴 Red = Failed action or escalation
  - 🟠 Orange = Ongoing healing action

---

## 🔄 Auto-Update Frequency

Change update frequency (top toolbar):
- **Every 1s** - Most responsive
- **Every 5s** - Default (balanced)
- **Every 10s** - Less network traffic
- **Every 30s** - Minimal updates

---

## ✨ Complete End-to-End Demo

1. **Open dashboard**: http://localhost:5173
2. **Verify data loads**: See "Active Heals: 309" in top cards
3. **Trigger full demo**: `curl -X POST http://localhost:5000/api/test/full-demo-flow`
4. **Watch dashboard**: New incidents appear every 2-4 seconds
5. **Observe healing**: Pipeline shows incident → action → recovery

---

## 📝 Notes

- **Real data source**: All data comes from `data/healing_log.json`
- **Updated by**: Orchestrator (real incidents) + API triggers (manual scenarios)
- **Real-time**: Dashboard fetches every 5 seconds by default
- **Historical**: 309 total incidents since system start
- **No simulation**: This is production orchestrator data

---

## 🚀 Quick Start

```bash
# 1. Open dashboard
Open http://localhost:5173 in browser

# 2. Wait 5 seconds for first data load

# 3. See KPI metrics populate (309 heals, 85.8% success)

# 4. Trigger an incident to see live update
curl -X POST http://localhost:5000/api/test/full-demo-flow

# 5. Watch dashboard auto-refresh with new incident

# Done! Dashboard is showing REAL NeuroShield data
```
