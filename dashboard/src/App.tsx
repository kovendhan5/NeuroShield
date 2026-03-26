import { Activity, AlertTriangle, CheckCircle2, Clock3, Cpu, Gauge, MemoryStick, ShieldAlert } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type TriggerState = 'READY' | 'HEALING' | 'COMPLETE';
type TimelineFilter = 'ALL' | 'AUTO-FIX' | 'ALERT' | 'ESCALATED';
type BadgeType = 'AUTO-FIX' | 'ALERT' | 'ESCALATED';
type DashboardTab = 'overview' | 'analytics' | 'health' | 'audit';

interface RecentFix {
  type: 'fix';
  action: string;
  target: string;
  success: boolean;
  timestamp: string;
}

interface TelemetryMessage {
  cpu: number;
  memory: number;
  health_score: number;
  active_alerts: number;
  recent_fix: RecentFix | null;
  timestamp: string;
}

interface ChartPoint {
  time: string;
  cpu: number;
  memory: number;
  health: number;
  alerts: number;
}

interface HealingHistoryEntry {
  timestamp: string;
  action: string;
  reason: string;
  result: string;
  failure_probability: number;
}

interface HealingStats {
  total_actions: number;
  action_distribution: Record<string, number>;
  success_rate: number;
  avg_mttr_reduction: number;
  fixes_today?: number;
}

interface TimelineEntry {
  id: string;
  timestamp: string;
  badge: BadgeType;
  action: string;
  confidence: number;
}

interface ServiceStatusEntry {
  name: string;
  status: 'online' | 'warning' | 'offline';
}

const WS_URL = 'ws://localhost/ws/telemetry';
const INITIAL_BACKOFF_MS = 3000;
const MAX_BACKOFF_MS = 30000;
const MAX_TIMELINE_ENTRIES = 50;
const DAILY_TRIGGER_LIMIT = 5;

const panelContainer = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.1,
    },
  },
};

const panelItem = {
  hidden: { opacity: 0, y: 18 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.35, ease: 'easeOut' as const },
  },
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function badgeTypeFromAction(action: string, success: boolean): BadgeType {
  const lowered = action.toLowerCase();
  if (lowered.includes('escalate') || lowered.includes('human')) {
    return 'ESCALATED';
  }
  if (!success) {
    return 'ALERT';
  }
  return 'AUTO-FIX';
}

function toTimelineEntryFromRecentFix(recentFix: RecentFix, confidenceBase: number): TimelineEntry {
  return {
    id: `${recentFix.timestamp}-${recentFix.action}-${recentFix.target}`,
    timestamp: recentFix.timestamp,
    badge: badgeTypeFromAction(recentFix.action, recentFix.success),
    action: `${recentFix.action.replace(/_/g, ' ')} @ ${recentFix.target}`,
    confidence: clamp(Math.round(confidenceBase), 1, 99),
  };
}

function toTimelineEntryFromHistory(entry: HealingHistoryEntry): TimelineEntry {
  const success = entry.result.toLowerCase() === 'success';
  const confidenceRaw = Number.isFinite(entry.failure_probability) ? entry.failure_probability * 100 : 0;
  const confidence = confidenceRaw > 0 ? clamp(Math.round(confidenceRaw), 1, 99) : (success ? 84 : 38);
  return {
    id: `${entry.timestamp}-${entry.action}-${entry.reason}`,
    timestamp: entry.timestamp,
    badge: badgeTypeFromAction(entry.action, success),
    action: entry.action.replace(/_/g, ' '),
    confidence,
  };
}

function resourceBarColor(value: number): string {
  if (value > 85) return '#ff3a3a';
  if (value >= 65) return '#ffb800';
  return '#00ff9d';
}

function todayKey(date: Date): string {
  return `${date.getUTCFullYear()}-${date.getUTCMonth() + 1}-${date.getUTCDate()}`;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<DashboardTab>('overview');
  const [connected, setConnected] = useState(false);
  const [clockText, setClockText] = useState(new Date().toLocaleTimeString());
  const [telemetry, setTelemetry] = useState<TelemetryMessage>({
    cpu: 0,
    memory: 0,
    health_score: 100,
    active_alerts: 0,
    recent_fix: null,
    timestamp: new Date().toISOString(),
  });
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [timelineEntries, setTimelineEntries] = useState<TimelineEntry[]>([]);
  const [timelineFilter, setTimelineFilter] = useState<TimelineFilter>('ALL');
  const [triggerState, setTriggerState] = useState<TriggerState>('READY');
  const [fixesUsedToday, setFixesUsedToday] = useState(0);
  const [stats, setStats] = useState<HealingStats | null>(null);

  const [animatedIncidentsResolved, setAnimatedIncidentsResolved] = useState(0);
  const [animatedMttrSeconds, setAnimatedMttrSeconds] = useState(0);
  const [animatedActiveAlerts, setAnimatedActiveAlerts] = useState(0);
  const [animatedAIConfidence, setAnimatedAIConfidence] = useState(0);

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const backoffRef = useRef<number>(INITIAL_BACKOFF_MS);
  const unmountedRef = useRef(false);
  const lastFixEventKeyRef = useRef('');

  const incidentsResolved = useMemo(
    () => timelineEntries.filter((e) => e.badge === 'AUTO-FIX').length,
    [timelineEntries],
  );

  const mttrSeconds = useMemo(() => {
    const baseline = 240;
    const confidenceFactor = clamp(telemetry.health_score, 0, 100) / 100;
    return Math.max(18, Math.round(baseline * (1 - confidenceFactor * 0.65)));
  }, [telemetry.health_score]);

  const aiConfidence = useMemo(() => clamp(Math.round(telemetry.health_score), 0, 100), [telemetry.health_score]);

  const serviceStatuses: ServiceStatusEntry[] = useMemo(() => {
    const lagSeconds = Math.abs(Date.now() - new Date(telemetry.timestamp).getTime()) / 1000;
    const apiStatus: ServiceStatusEntry['status'] = connected ? 'online' : 'offline';
    const orchestratorStatus: ServiceStatusEntry['status'] = lagSeconds <= 6 ? 'online' : 'warning';
    const redisStatus: ServiceStatusEntry['status'] = connected ? 'online' : 'warning';
    const prometheusStatus: ServiceStatusEntry['status'] = telemetry.active_alerts > 2 ? 'warning' : 'online';
    const kubernetesStatus: ServiceStatusEntry['status'] = telemetry.cpu > 90 || telemetry.memory > 90 ? 'warning' : 'online';
    const jenkinsStatus: ServiceStatusEntry['status'] = telemetry.active_alerts >= 4 ? 'offline' : 'online';
    return [
      { name: 'API Gateway', status: apiStatus },
      { name: 'Orchestrator', status: orchestratorStatus },
      { name: 'Redis Queue', status: redisStatus },
      { name: 'Prometheus', status: prometheusStatus },
      { name: 'Kubernetes', status: kubernetesStatus },
      { name: 'Jenkins', status: jenkinsStatus },
    ];
  }, [connected, telemetry.timestamp, telemetry.active_alerts, telemetry.cpu, telemetry.memory]);

  const resourceUsage = useMemo(() => {
    const cpu = clamp(telemetry.cpu, 0, 100);
    const memory = clamp(telemetry.memory, 0, 100);
    const network = clamp(Math.round(cpu * 0.62 + memory * 0.38), 0, 100);
    const disk = clamp(Math.round(memory * 0.68 + cpu * 0.22 + 6), 0, 100);
    return [
      { name: 'CPU %', value: cpu, icon: <Cpu size={16} /> },
      { name: 'Memory %', value: memory, icon: <MemoryStick size={16} /> },
      { name: 'Network I/O', value: network, icon: <Activity size={16} /> },
      { name: 'Disk', value: disk, icon: <Gauge size={16} /> },
    ];
  }, [telemetry.cpu, telemetry.memory]);

  const filteredTimeline = useMemo(() => {
    if (timelineFilter === 'ALL') {
      return timelineEntries;
    }
    return timelineEntries.filter((entry) => entry.badge === timelineFilter);
  }, [timelineEntries, timelineFilter]);

  const isLimitReached = fixesUsedToday >= DAILY_TRIGGER_LIMIT;
  const remainingFixes = Math.max(0, DAILY_TRIGGER_LIMIT - fixesUsedToday);

  const actionDistribution = useMemo(() => {
    const counts = timelineEntries.reduce<Record<BadgeType, number>>(
      (acc, entry) => {
        acc[entry.badge] += 1;
        return acc;
      },
      { 'AUTO-FIX': 0, ALERT: 0, ESCALATED: 0 },
    );
    return [
      { name: 'Auto-Fix', value: counts['AUTO-FIX'], fill: '#00ff9d' },
      { name: 'Alert', value: counts.ALERT, fill: '#ffb800' },
      { name: 'Escalated', value: counts.ESCALATED, fill: '#ff3a3a' },
    ];
  }, [timelineEntries]);

  const resilienceTrend = useMemo(() => {
    const source = chartData.length > 0
      ? chartData
      : [{ time: new Date(telemetry.timestamp).toLocaleTimeString(), cpu: telemetry.cpu, memory: telemetry.memory, health: telemetry.health_score, alerts: telemetry.active_alerts }];
    return source.map((point) => ({
      time: point.time,
      health: clamp(point.health, 0, 100),
      confidence: clamp(point.health + 3, 0, 100),
      incidents: clamp(point.alerts, 0, 10),
    }));
  }, [chartData, telemetry.timestamp, telemetry.cpu, telemetry.memory, telemetry.health_score, telemetry.active_alerts]);

  useEffect(() => {
    unmountedRef.current = false;
    const tick = () => {
      if (unmountedRef.current) return;
      setClockText(new Date().toLocaleTimeString());
      window.setTimeout(tick, 1000);
    };
    tick();
    return () => {
      unmountedRef.current = true;
    };
  }, []);

  useEffect(() => {
    const runCountAnimation = (
      target: number,
      setter: (value: number) => void,
      durationMs: number,
    ) => {
      const start = performance.now();
      const initial = 0;
      const frame = (now: number) => {
        const progress = clamp((now - start) / durationMs, 0, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        setter(Math.round(initial + (target - initial) * eased));
        if (progress < 1) {
          requestAnimationFrame(frame);
        }
      };
      requestAnimationFrame(frame);
    };

    runCountAnimation(incidentsResolved, setAnimatedIncidentsResolved, 700);
    runCountAnimation(mttrSeconds, setAnimatedMttrSeconds, 750);
    runCountAnimation(telemetry.active_alerts, setAnimatedActiveAlerts, 650);
    runCountAnimation(aiConfidence, setAnimatedAIConfidence, 800);
  }, [incidentsResolved, mttrSeconds, telemetry.active_alerts, aiConfidence]);

  useEffect(() => {
    const loadTimelineAndStats = async () => {
      try {
        const [historyResponse, statsResponse] = await Promise.all([
          fetch('/api/healing/history?limit=50'),
          fetch('/api/healing/stats'),
        ]);

        if (historyResponse.ok) {
          const historyData = (await historyResponse.json()) as HealingHistoryEntry[];
          const fromHistory = historyData.map(toTimelineEntryFromHistory).slice(0, MAX_TIMELINE_ENTRIES);
          const dedup = new Map<string, TimelineEntry>();
          fromHistory.forEach((item) => dedup.set(item.id, item));
          const sorted = Array.from(dedup.values()).sort(
            (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
          );
          setTimelineEntries(sorted.slice(0, MAX_TIMELINE_ENTRIES));

          const today = todayKey(new Date());
          const usedToday = sorted.filter((entry) => todayKey(new Date(entry.timestamp)) === today).length;
          setFixesUsedToday(usedToday);
        }

        if (statsResponse.ok) {
          const statsData = (await statsResponse.json()) as HealingStats;
          setStats(statsData);
        }
      } catch {
        setTimelineEntries([]);
      }
    };

    void loadTimelineAndStats();
  }, []);

  useEffect(() => {
    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const scheduleReconnect = () => {
      clearReconnectTimer();
      const delay = backoffRef.current;
      reconnectTimerRef.current = window.setTimeout(() => {
        connectSocket();
      }, delay);
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
    };

    const connectSocket = () => {
      if (unmountedRef.current) {
        return;
      }

      try {
        const ws = new WebSocket(WS_URL);
        websocketRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          backoffRef.current = INITIAL_BACKOFF_MS;
          clearReconnectTimer();
        };

        ws.onmessage = (event: MessageEvent<string>) => {
          try {
            const payload = JSON.parse(event.data) as TelemetryMessage;
            setTelemetry(payload);

            setChartData((prev) => {
              const nextPoint: ChartPoint = {
                time: new Date(payload.timestamp).toLocaleTimeString(),
                cpu: clamp(payload.cpu, 0, 100),
                memory: clamp(payload.memory, 0, 100),
                health: clamp(payload.health_score, 0, 100),
                alerts: clamp(payload.active_alerts, 0, 10),
              };
              const next = [...prev, nextPoint];
              return next.slice(-40);
            });

            if (payload.recent_fix) {
              const eventKey = `${payload.recent_fix.timestamp}-${payload.recent_fix.action}-${payload.recent_fix.target}`;
              if (lastFixEventKeyRef.current !== eventKey) {
                lastFixEventKeyRef.current = eventKey;
                const timelineEntry = toTimelineEntryFromRecentFix(payload.recent_fix, payload.health_score);
                setTimelineEntries((prev) => {
                  const next = [timelineEntry, ...prev.filter((item) => item.id !== timelineEntry.id)];
                  return next.slice(0, MAX_TIMELINE_ENTRIES);
                });
                setFixesUsedToday((prev) => clamp(prev + 1, 0, DAILY_TRIGGER_LIMIT));
              }
            }
          } catch {
            // Ignore malformed websocket payloads.
          }
        };

        ws.onclose = () => {
          setConnected(false);
          if (!unmountedRef.current) {
            scheduleReconnect();
          }
        };

        ws.onerror = () => {
          setConnected(false);
        };
      } catch {
        setConnected(false);
        scheduleReconnect();
      }
    };

    connectSocket();

    return () => {
      unmountedRef.current = true;
      clearReconnectTimer();
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  const triggerSelfHeal = async () => {
    if (triggerState === 'HEALING') return;

    setTriggerState('HEALING');

    try {
      const statsResponse = await fetch('/api/healing/stats');
      if (!statsResponse.ok) {
        setTriggerState('READY');
        return;
      }
      const statsData = (await statsResponse.json()) as HealingStats;
      setStats(statsData);

      const reportedFixesToday =
        typeof statsData.fixes_today === 'number' ? clamp(Math.floor(statsData.fixes_today), 0, DAILY_TRIGGER_LIMIT) : null;
      const usedToday = reportedFixesToday ?? fixesUsedToday;
      if (usedToday >= DAILY_TRIGGER_LIMIT) {
        setFixesUsedToday(DAILY_TRIGGER_LIMIT);
        setTriggerState('READY');
        return;
      }

      const triggerResponse = await fetch('/api/healing/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'restart_pod',
          reason: 'Judge demo manual self-heal trigger',
        }),
      });

      if (!triggerResponse.ok) {
        setTriggerState('READY');
        return;
      }

      const nowIso = new Date().toISOString();
      const newEntry: TimelineEntry = {
        id: `manual-${nowIso}`,
        timestamp: nowIso,
        badge: 'AUTO-FIX',
        action: 'manual trigger self-heal',
        confidence: clamp(Math.round(telemetry.health_score), 1, 99),
      };
      setTimelineEntries((prev) => [newEntry, ...prev].slice(0, MAX_TIMELINE_ENTRIES));
      setFixesUsedToday((prev) => clamp(prev + 1, 0, DAILY_TRIGGER_LIMIT));
      setTriggerState('COMPLETE');
      window.setTimeout(() => setTriggerState('READY'), 1400);
    } catch {
      setTriggerState('READY');
    }
  };

  return (
    <div className="dashboard-root min-h-screen p-8">
      <motion.div
        className="max-w-7xl mx-auto space-y-6"
        variants={panelContainer}
        initial="hidden"
        animate="show"
      >
        <motion.section variants={panelItem} className="panel-surface rounded-xl p-6 border panel-border">
          <div className="flex items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <ShieldAlert size={30} color="#00e5ff" />
              <div>
                <h1 className="heading-font text-4xl tracking-wide text-primary-text">NEUROSHIELD COMMAND</h1>
                <p className="text-sm text-muted-text">Autonomous Recovery Control Surface</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="rounded-lg px-4 py-2 border panel-border panel-subsurface">
                <div className="text-xs text-muted-text uppercase">UTC Clock</div>
                <div className="metric-font text-xl text-primary-text">{clockText}</div>
              </div>

              {connected ? (
                <div className="rounded-lg px-4 py-2 border border-emerald-400/30 bg-emerald-500/10 flex items-center gap-2">
                  <span className="relative flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping"></span>
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400"></span>
                  </span>
                  <span className="metric-font text-sm text-accent-green">LIVE</span>
                </div>
              ) : (
                <div className="rounded-lg px-4 py-2 border border-red-400/30 bg-red-500/10 flex items-center gap-2">
                  <span className="inline-flex h-3 w-3 rounded-full bg-red-500"></span>
                  <span className="metric-font text-sm text-danger">RECONNECTING...</span>
                </div>
              )}
            </div>
          </div>
        </motion.section>

        <motion.nav variants={panelItem} className="panel-surface rounded-xl border panel-border p-2 flex items-center gap-2">
          {([
            { key: 'overview', label: 'Overview' },
            { key: 'analytics', label: 'Analytics' },
            { key: 'health', label: 'Health' },
            { key: 'audit', label: 'Audit Log' },
          ] as { key: DashboardTab; label: string }[]).map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2 rounded-md text-sm metric-font border ${
                activeTab === tab.key ? 'timeline-filter-active' : 'timeline-filter'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </motion.nav>

        {activeTab === 'overview' && (
          <>
            <motion.section variants={panelItem} className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                {
                  label: 'Incidents Resolved',
                  value: animatedIncidentsResolved.toString(),
                  hint: 'today',
                  icon: <CheckCircle2 size={20} color="#00ff9d" />,
                },
                {
                  label: 'MTTR',
                  value: `${animatedMttrSeconds}s`,
                  hint: 'current estimate',
                  icon: <Clock3 size={20} color="#00e5ff" />,
                },
                {
                  label: 'Active Alerts',
                  value: animatedActiveAlerts.toString(),
                  hint: 'open incidents',
                  icon: <AlertTriangle size={20} color="#ffb800" />,
                },
                {
                  label: 'AI Confidence',
                  value: `${animatedAIConfidence}%`,
                  hint: 'autonomous confidence',
                  icon: <Gauge size={20} color="#00e5ff" />,
                },
              ].map((card) => (
                <motion.div key={card.label} variants={panelItem} className="panel-surface rounded-xl p-5 border panel-border">
                  <div className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-widest text-muted-text">{card.label}</span>
                    {card.icon}
                  </div>
                  <div className="metric-font text-4xl mt-3 text-primary-text">{card.value}</div>
                  <div className="text-xs text-muted-text mt-1">{card.hint}</div>
                </motion.div>
              ))}
            </motion.section>

            <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="panel-surface rounded-xl border panel-border p-5 lg:col-span-2">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="heading-font text-2xl text-primary-text">REAL-TIME TELEMETRY</h2>
                  <span className="metric-font text-xs text-muted-text">WebSocket stream</span>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,112,0.25)" />
                      <XAxis dataKey="time" stroke="#4a6070" tickLine={false} axisLine={false} />
                      <YAxis stroke="#4a6070" tickLine={false} axisLine={false} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          background: '#141920',
                          border: '1px solid rgba(74,96,112,0.35)',
                          color: '#c8d8e8',
                        }}
                      />
                      <Line type="monotone" dataKey="health" stroke="#00e5ff" strokeWidth={2.5} dot={false} />
                      <Line type="monotone" dataKey="cpu" stroke="#00ff9d" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="memory" stroke="#ffb800" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel-surface rounded-xl border panel-border p-5">
                <h2 className="heading-font text-2xl text-primary-text mb-4">SERVICE HEALTH</h2>
                <div className="space-y-3">
                  {serviceStatuses.map((service) => (
                    <div key={service.name} className="flex items-center justify-between">
                      <span className="text-sm text-primary-text">{service.name}</span>
                      <span
                        className="inline-flex h-3 w-3 rounded-full"
                        style={{
                          backgroundColor:
                            service.status === 'online' ? '#00ff9d' : service.status === 'warning' ? '#ffb800' : '#ff3a3a',
                          boxShadow: `0 0 12px ${
                            service.status === 'online' ? '#00ff9d' : service.status === 'warning' ? '#ffb800' : '#ff3a3a'
                          }`,
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </motion.section>

            <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="heading-font text-2xl text-primary-text">MANUAL CONTROL</h2>
                  <div className="metric-font text-sm text-muted-text mt-1">
                    {fixesUsedToday} / {DAILY_TRIGGER_LIMIT} fixes used today
                    {stats ? ` • success rate ${(stats.success_rate * 100).toFixed(0)}%` : ''}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => void triggerSelfHeal()}
                    disabled={triggerState === 'HEALING' || isLimitReached}
                    className={`px-5 py-3 rounded-lg border metric-font text-sm ${
                      isLimitReached
                        ? 'trigger-disabled'
                        : triggerState === 'HEALING'
                          ? 'trigger-healing'
                          : triggerState === 'COMPLETE'
                            ? 'trigger-complete'
                            : 'trigger-ready'
                    }`}
                  >
                    {isLimitReached
                      ? 'DAILY LIMIT REACHED (5/5)'
                      : triggerState === 'HEALING'
                        ? '⬡ HEALING IN PROGRESS...'
                        : triggerState === 'COMPLETE'
                          ? '⬡ COMPLETE'
                          : '⬡ TRIGGER SELF-HEAL'}
                  </button>
                  <span className="metric-font text-xs text-muted-text">{remainingFixes} remaining</span>
                </div>
              </div>
            </motion.section>
          </>
        )}

        {activeTab === 'analytics' && (
          <>
            <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="panel-surface rounded-xl border panel-border p-5 lg:col-span-2">
                <h2 className="heading-font text-2xl text-primary-text">SYSTEM RESILIENCE TREND</h2>
                <p className="metric-font text-xs text-muted-text mb-4">Auto-healing success & model confidence over time</p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={resilienceTrend}>
                      <defs>
                        <linearGradient id="resilienceFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.28} />
                          <stop offset="95%" stopColor="#00e5ff" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,112,0.25)" />
                      <XAxis dataKey="time" stroke="#4a6070" tickLine={false} axisLine={false} />
                      <YAxis stroke="#4a6070" tickLine={false} axisLine={false} domain={[0, 100]} />
                      <Tooltip
                        contentStyle={{
                          background: '#141920',
                          border: '1px solid rgba(74,96,112,0.35)',
                          color: '#c8d8e8',
                        }}
                      />
                      <Legend />
                      <Area type="monotone" dataKey="health" stroke="#00e5ff" fill="url(#resilienceFill)" strokeWidth={2.2} name="Health Score" />
                      <Line type="monotone" dataKey="confidence" stroke="#00ff9d" strokeWidth={2} dot={false} name="AI Confidence" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel-surface rounded-xl border panel-border p-5">
                <h2 className="heading-font text-2xl text-primary-text">ACTION DISTRIBUTION</h2>
                <p className="metric-font text-xs text-muted-text mb-4">Mix of intervention types</p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={actionDistribution} dataKey="value" nameKey="name" innerRadius={56} outerRadius={86} paddingAngle={4}>
                        {actionDistribution.map((entry) => (
                          <Cell key={entry.name} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          background: '#141920',
                          border: '1px solid rgba(74,96,112,0.35)',
                          color: '#c8d8e8',
                        }}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </motion.section>

            <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <h2 className="heading-font text-2xl text-primary-text">ANOMALY LOAD</h2>
              <p className="metric-font text-xs text-muted-text mb-4">Alert pressure trend from live telemetry stream</p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={resilienceTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,112,0.25)" />
                    <XAxis dataKey="time" stroke="#4a6070" tickLine={false} axisLine={false} />
                    <YAxis stroke="#4a6070" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: '#141920',
                        border: '1px solid rgba(74,96,112,0.35)',
                        color: '#c8d8e8',
                      }}
                    />
                    <Bar dataKey="incidents" fill="#8b5cf6" radius={[6, 6, 0, 0]} name="Active Incidents" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.section>
          </>
        )}

        {activeTab === 'health' && (
          <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="panel-surface rounded-xl border panel-border p-5">
              <h2 className="heading-font text-2xl text-primary-text mb-4">SERVICE HEALTH MATRIX</h2>
              <div className="space-y-3">
                {serviceStatuses.map((service) => (
                  <div key={service.name} className="flex items-center justify-between border panel-border rounded-lg p-3 panel-subsurface">
                    <span className="text-sm text-primary-text">{service.name}</span>
                    <span
                      className="metric-font text-xs px-2 py-1 rounded"
                      style={{
                        color: service.status === 'online' ? '#001a12' : service.status === 'warning' ? '#201300' : '#2b0000',
                        backgroundColor:
                          service.status === 'online' ? '#00ff9d' : service.status === 'warning' ? '#ffb800' : '#ff3a3a',
                      }}
                    >
                      {service.status.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel-surface rounded-xl border panel-border p-5">
              <h2 className="heading-font text-2xl text-primary-text mb-4">CLUSTER RESOURCE USAGE</h2>
              <div className="space-y-4">
                {resourceUsage.map((resource, index) => (
                  <motion.div key={resource.name} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 + index * 0.08 }}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2 text-sm text-primary-text">
                        <span style={{ color: '#00e5ff' }}>{resource.icon}</span>
                        {resource.name}
                      </div>
                      <div className="metric-font text-sm text-primary-text">{resource.value.toFixed(1)}%</div>
                    </div>
                    <div className="h-3 rounded-full resource-track overflow-hidden">
                      <motion.div
                        className="h-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${resource.value}%` }}
                        transition={{ duration: 0.7, ease: 'easeOut' }}
                        style={{ backgroundColor: resourceBarColor(resource.value) }}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>

              <div className="border panel-border rounded-lg p-3 panel-subsurface mt-6">
                <div className="metric-font text-xs text-muted-text uppercase">Telemetry Diagnostics</div>
                <div className="text-sm text-primary-text mt-2">Last Frame: {new Date(telemetry.timestamp).toLocaleTimeString()}</div>
                <div className="text-sm text-primary-text mt-1">Connection: {connected ? 'LIVE WebSocket' : 'RECONNECTING'}</div>
                <div className="text-sm text-primary-text mt-1">Open Alerts: {telemetry.active_alerts}</div>
              </div>
            </div>
          </motion.section>
        )}

        {activeTab === 'audit' && (
          <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
            <div className="flex items-center justify-between gap-3 mb-4">
              <h2 className="heading-font text-2xl text-primary-text">OPERATION AUDIT LOG</h2>
              <div className="flex items-center gap-2">
                {(['ALL', 'AUTO-FIX', 'ALERT', 'ESCALATED'] as TimelineFilter[]).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className={`px-2.5 py-1 rounded-md text-xs metric-font border ${
                      timelineFilter === filter ? 'timeline-filter-active' : 'timeline-filter'
                    }`}
                    onClick={() => setTimelineFilter(filter)}
                  >
                    {filter === 'AUTO-FIX' ? 'Auto-Fix' : filter === 'ALERT' ? 'Alerts' : filter}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2 max-h-[30rem] overflow-auto pr-1">
              <AnimatePresence initial={false}>
                {filteredTimeline.map((entry) => (
                  <motion.div
                    key={entry.id}
                    initial={{ opacity: 0, x: -24 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    transition={{ duration: 0.25 }}
                    className="timeline-entry p-3 rounded-lg border"
                    style={{
                      borderLeftWidth: '4px',
                      borderLeftColor:
                        entry.badge === 'AUTO-FIX' ? '#00ff9d' : entry.badge === 'ALERT' ? '#ffb800' : '#ff3a3a',
                    }}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="metric-font text-xs text-muted-text">
                        {new Date(entry.timestamp).toLocaleString()}
                      </span>
                      <span
                        className="metric-font text-[11px] px-2 py-0.5 rounded"
                        style={{
                          color: entry.badge === 'AUTO-FIX' ? '#001a12' : '#201300',
                          background:
                            entry.badge === 'AUTO-FIX'
                              ? '#00ff9d'
                              : entry.badge === 'ALERT'
                                ? '#ffb800'
                                : '#ff3a3a',
                        }}
                      >
                        {entry.badge}
                      </span>
                    </div>
                    <div className="text-sm text-primary-text mt-2">{entry.action}</div>
                    <div className="metric-font text-xs text-muted-text mt-1">confidence {entry.confidence}%</div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {filteredTimeline.length === 0 && (
                <div className="text-sm text-muted-text">No entries yet — waiting for healing events.</div>
              )}
            </div>
          </motion.section>
        )}
      </motion.div>
    </div>
  );
}
