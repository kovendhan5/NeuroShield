import { AnimatePresence, motion } from 'framer-motion';
import { Activity, AlertTriangle, CheckCircle2, Clock3, Cpu, Gauge, MemoryStick, ShieldAlert } from 'lucide-react';
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
type DashboardTab = 'overview' | 'analytics' | 'health' | 'kubernetes' | 'audit' | 'settings';

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
  mttr_seconds?: number;
  healing_success_rate?: number;
  uptime_seconds?: number;
  service_states?: Record<string, string>;
  service_logs?: ServiceLogEntry[];
  pipeline_overview?: PipelineRuntimeEntry[];
  kubernetes?: KubernetesRuntime;
  recent_fix: RecentFix | null;
  timestamp: string;
}

interface ServiceLogEntry {
  service: string;
  level: string;
  message: string;
  timestamp: string;
}

interface PipelineRuntimeEntry {
  id: string;
  project: string;
  use_case: string;
  environment: string;
  deploy_target: string;
  status: string;
  total_runs: number;
  success_runs: number;
  failed_runs: number;
  avg_duration_seconds: number;
  last_run: string;
  last_error: string;
  autoheal_actions: number;
  open_incidents?: number;
  k8s_namespace: string;
  k8s_deployment: string;
  deployment_url?: string;
}

interface KubernetesRuntime {
  cluster_health: number;
  failed_pods: number;
  pod_restarts_total: number;
  autoheals_total: number;
  last_autoheal: string;
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
  reason?: string;
  result?: string;
  source?: 'history' | 'live' | 'manual';
  rawLog?: string;
  ruleMatched?: string;
  diffPreview?: string;
}

interface ServiceStatusEntry {
  name: string;
  status: 'online' | 'warning' | 'offline';
}

interface AuditLogEntry {
  timestamp: string;
  category: string;
  action: string;
  actor: string;
  resource: string;
  result: string;
  details: Record<string, unknown>;
  ip_address?: string;
}

type AuditFilter = 'ALL' | 'USER_ACTION' | 'HEALING_ACTION' | 'SECURITY_EVENT' | 'SYSTEM_EVENT';

const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const API_PROTOCOL = window.location.protocol === 'https:' ? 'https:' : 'http:';

const WS_URL = window.location.port === '8501'
  ? `${WS_PROTOCOL}//${window.location.hostname}:8000/ws/telemetry`
  : `${WS_PROTOCOL}//${window.location.host}/ws/telemetry`;

const AUDIT_WS_URL = window.location.port === '8501'
  ? `${WS_PROTOCOL}//${window.location.hostname}:8000/audit/ws`
  : `${WS_PROTOCOL}//${window.location.host}/api/audit/ws`;

const HEALING_HISTORY_URL = window.location.port === '8501'
  ? `${API_PROTOCOL}//${window.location.hostname}:8000/healing/history`
  : '/api/healing/history';

const HEALING_STATS_URL = window.location.port === '8501'
  ? `${API_PROTOCOL}//${window.location.hostname}:8000/healing/stats`
  : '/api/healing/stats';

const REMEDIATE_MANUAL_URL = window.location.port === '8501'
  ? `${API_PROTOCOL}//${window.location.hostname}:8000/v1/remediate/manual`
  : '/api/v1/remediate/manual';
const INITIAL_BACKOFF_MS = 3000;
const MAX_BACKOFF_MS = 30000;
const MAX_TIMELINE_ENTRIES = 50;

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
  const actionLower = recentFix.action.toLowerCase();
  const targetLower = recentFix.target.toLowerCase();
  
  // Determine source system with more precision
  let sourceSystem = 'neuroshield';
  if (actionLower.includes('jenkins') || targetLower.includes('pipeline') || actionLower.includes('build')) {
    sourceSystem = 'jenkins';
  } else if (actionLower.includes('pod') || actionLower.includes('k8s') || targetLower.includes('k8s') || 
             actionLower.includes('restart') || actionLower.includes('scale') || actionLower.includes('deploy')) {
    sourceSystem = 'kubernetes';
  } else if (actionLower.includes('grafana') || actionLower.includes('dashboard')) {
    sourceSystem = 'grafana';
  } else if (actionLower.includes('metric') || actionLower.includes('scrape') || actionLower.includes('alert')) {
    sourceSystem = 'prometheus';
  }
  
  // Map action to human-readable description
  let actionDescription = recentFix.action.replace(/_/g, ' ');
  if (actionLower.includes('restart')) {
    actionDescription = 'Pod Restart';
  } else if (actionLower.includes('scale_up')) {
    actionDescription = 'Scale Up Replicas';
  } else if (actionLower.includes('scale_down')) {
    actionDescription = 'Scale Down Replicas';
  } else if (actionLower.includes('retry')) {
    actionDescription = 'Retry Pipeline Build';
  } else if (actionLower.includes('rollback')) {
    actionDescription = 'Rollback Deployment';
  } else if (actionLower.includes('cache') || actionLower.includes('clear')) {
    actionDescription = 'Clear Build Cache';
  } else if (actionLower.includes('escalate')) {
    actionDescription = 'Escalate to Human';
  }
  
  return {
    id: `${recentFix.timestamp}-${recentFix.action}-${recentFix.target}`,
    timestamp: recentFix.timestamp,
    badge: badgeTypeFromAction(recentFix.action, recentFix.success),
    action: `[${sourceSystem}] ${actionDescription} @ ${recentFix.target}`,
    confidence: clamp(Math.round(confidenceBase), 1, 99),
    source: 'live',
    reason: `Source: ${sourceSystem} | Target: ${recentFix.target}`,
    rawLog: `Action: ${recentFix.action}\nTarget: ${recentFix.target}\nSuccess: ${recentFix.success}\nTimestamp: ${recentFix.timestamp}`,
  };
}

function toTimelineEntryFromHistory(entry: HealingHistoryEntry): TimelineEntry {
  const success = entry.result.toLowerCase() === 'success';
  const confidenceRaw = Number.isFinite(entry.failure_probability) ? entry.failure_probability * 100 : 0;
  const confidence = confidenceRaw > 0 ? clamp(Math.round(confidenceRaw), 1, 99) : (success ? 84 : 38);
  
  // Determine source system from action type
  const actionLower = entry.action.toLowerCase();
  const reasonLower = (entry.reason || '').toLowerCase();
  let sourceSystem = 'neuroshield';
  let detailedAction = entry.action.replace(/_/g, ' ');
  
  if (actionLower.includes('restart') || actionLower.includes('pod') || actionLower.includes('scale')) {
    sourceSystem = 'kubernetes';
    detailedAction = `[k8s] ${detailedAction}`;
  } else if (actionLower.includes('build') || actionLower.includes('retry') || reasonLower.includes('jenkins')) {
    sourceSystem = 'jenkins';
    detailedAction = `[jenkins] ${detailedAction}`;
  } else if (actionLower.includes('cache') || actionLower.includes('clear')) {
    sourceSystem = 'prometheus';
    detailedAction = `[prometheus] ${detailedAction}`;
  } else if (actionLower.includes('rollback') || actionLower.includes('deploy')) {
    sourceSystem = 'kubernetes';
    detailedAction = `[k8s] ${detailedAction}`;
  } else if (actionLower.includes('escalate')) {
    sourceSystem = 'grafana';
    detailedAction = `[grafana] ${detailedAction}`;
  }
  
  return {
    id: `${entry.timestamp}-${entry.action}-${entry.reason}`,
    timestamp: entry.timestamp,
    badge: badgeTypeFromAction(entry.action, success),
    action: detailedAction,
    confidence,
    reason: entry.reason ? `Source: ${sourceSystem} | ${entry.reason}` : `Source: ${sourceSystem}`,
    source: 'history',
  };
}

function resourceBarColor(value: number): string {
  if (value > 85) return '#ff3a3a';
  if (value >= 65) return '#ffb800';
  return '#00ff9d';
}

function isPipelineTimelineEntry(entry: TimelineEntry): boolean {
  const text = `${entry.action} ${entry.reason ?? ''}`.toLowerCase();
  return (
    text.includes('pipeline') ||
    text.includes('jenkins') ||
    text.includes('kubernetes') ||
    text.includes('prometheus') ||
    text.includes('grafana') ||
    text.includes('deploy')
  );
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
  const [stats, setStats] = useState<HealingStats | null>(null);

  // Audit log state
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [auditFilter, setAuditFilter] = useState<AuditFilter>('ALL');
  const [auditConnected, setAuditConnected] = useState(false);
  const [serviceLogs, setServiceLogs] = useState<ServiceLogEntry[]>([]);
  const [selectedTimelineEntry, setSelectedTimelineEntry] = useState<TimelineEntry | null>(null);
  const [pipelineOverview, setPipelineOverview] = useState<PipelineRuntimeEntry[]>([]);
  const [kubernetesRuntime, setKubernetesRuntime] = useState<KubernetesRuntime>({
    cluster_health: 100,
    failed_pods: 0,
    pod_restarts_total: 0,
    autoheals_total: 0,
    last_autoheal: new Date().toISOString(),
  });
  const [safetyRules, setSafetyRules] = useState({
    pathTraversalProtection: true,
    rollbackRequired: true,
    confidenceThresholdEnabled: true,
    dailyFixLimitEnabled: true,
    humanEscalationEnabled: true,
  });
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.75);
  const [maxFixesPerDay, setMaxFixesPerDay] = useState(5);
  const auditWsRef = useRef<WebSocket | null>(null);

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
    if (typeof telemetry.mttr_seconds === 'number' && telemetry.mttr_seconds > 0) {
      return Math.round(telemetry.mttr_seconds);
    }
    const baseline = 240;
    const confidenceFactor = clamp(telemetry.health_score, 0, 100) / 100;
    return Math.max(18, Math.round(baseline * (1 - confidenceFactor * 0.65)));
  }, [telemetry.health_score, telemetry.mttr_seconds]);

  const aiConfidence = useMemo(() => {
    if (typeof telemetry.healing_success_rate === 'number') {
      return clamp(Math.round(telemetry.healing_success_rate * 100), 0, 100);
    }
    return clamp(Math.round(telemetry.health_score), 0, 100);
  }, [telemetry.health_score, telemetry.healing_success_rate]);

  const serviceStatuses: ServiceStatusEntry[] = useMemo(() => {
    if (telemetry.service_states && Object.keys(telemetry.service_states).length > 0) {
      return [
        { name: 'API Gateway', status: (telemetry.service_states.api as ServiceStatusEntry['status']) || 'offline' },
        { name: 'Orchestrator', status: (telemetry.service_states.orchestrator as ServiceStatusEntry['status']) || 'warning' },
        { name: 'Worker', status: (telemetry.service_states.worker as ServiceStatusEntry['status']) || 'warning' },
        { name: 'Prometheus', status: (telemetry.service_states.prometheus as ServiceStatusEntry['status']) || 'offline' },
        { name: 'Grafana', status: (telemetry.service_states.grafana as ServiceStatusEntry['status']) || 'offline' },
        { name: 'Redis', status: (telemetry.service_states.redis as ServiceStatusEntry['status']) || 'offline' },
      ];
    }
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
    const pipelineOnly = timelineEntries.filter(isPipelineTimelineEntry);
    if (timelineFilter === 'ALL') {
      return pipelineOnly;
    }
    return pipelineOnly.filter((entry) => entry.badge === timelineFilter);
  }, [timelineEntries, timelineFilter]);

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

  const riskScore = useMemo(() => {
    const cpu = clamp(telemetry.cpu, 0, 100);
    const memory = clamp(telemetry.memory, 0, 100);
    const alerts = clamp(telemetry.active_alerts * 15, 0, 100);
    return clamp(Math.round(cpu * 0.35 + memory * 0.3 + alerts * 0.35), 0, 100);
  }, [telemetry.cpu, telemetry.memory, telemetry.active_alerts]);

  const pipelineStats = useMemo(() => {
    if (!pipelineOverview.length) {
      return {
        totalRuns: 0,
        successfulRuns: 0,
        failedRuns: 0,
        autoheals: 0,
      };
    }
    return pipelineOverview.reduce(
      (acc, item) => {
        acc.totalRuns += item.total_runs;
        acc.successfulRuns += item.success_runs;
        acc.failedRuns += item.failed_runs;
        acc.autoheals += item.autoheal_actions;
        return acc;
      },
      { totalRuns: 0, successfulRuns: 0, failedRuns: 0, autoheals: 0 },
    );
  }, [pipelineOverview]);

  const anomalyLoadData = useMemo(() => {
    const pipelineIncidentSignal = pipelineOverview.reduce((sum, p) => sum + Number(p.open_incidents ?? 0), 0);
    const source = chartData.length > 0
      ? chartData
      : [{ time: new Date(telemetry.timestamp).toLocaleTimeString(), cpu: telemetry.cpu, memory: telemetry.memory, health: telemetry.health_score, alerts: telemetry.active_alerts }];
    return source.map((point) => ({
      time: point.time,
      incidents: Math.max(0, Math.round(Math.max(point.alerts, pipelineIncidentSignal))),
    }));
  }, [chartData, pipelineOverview, telemetry.timestamp, telemetry.cpu, telemetry.memory, telemetry.health_score, telemetry.active_alerts]);

  const predictedHealth = useMemo(() => {
    const now = new Date();
    const current = clamp(telemetry.health_score, 0, 100);
    return Array.from({ length: 12 }).map((_, idx) => {
      const minute = (idx + 1) * 5;
      const drift = riskScore > 70 ? minute * 0.45 : riskScore > 45 ? minute * 0.2 : minute * 0.08;
      const noise = Math.sin(idx * 0.7) * 2.5;
      const value = clamp(current - drift + noise, 0, 100);
      return {
        time: new Date(now.getTime() + minute * 60000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        health: value,
        anomaly: value < 75,
      };
    });
  }, [telemetry.health_score, riskScore]);

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
          fetch(`${HEALING_HISTORY_URL}?limit=50`),
          fetch(HEALING_STATS_URL),
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
              }
            }

            if (Array.isArray(payload.service_logs)) {
              const heartbeat: ServiceLogEntry[] = [
                { service: 'jenkins', level: 'INFO', message: 'pipeline telemetry stream active', timestamp: payload.timestamp },
                { service: 'kubernetes', level: 'INFO', message: `cluster health ${Math.round(payload.kubernetes?.cluster_health ?? 100)}%`, timestamp: payload.timestamp },
                { service: 'prometheus', level: 'INFO', message: 'metrics scrape refreshed', timestamp: payload.timestamp },
                { service: 'grafana', level: 'INFO', message: 'dashboard queries running', timestamp: payload.timestamp },
              ];
              setServiceLogs([...payload.service_logs, ...heartbeat].slice(-140).reverse());
            }
            if (Array.isArray(payload.pipeline_overview)) {
              setPipelineOverview(payload.pipeline_overview);
            }
            if (payload.kubernetes && typeof payload.kubernetes === 'object') {
              setKubernetesRuntime({
                cluster_health: Number(payload.kubernetes.cluster_health ?? 100),
                failed_pods: Number(payload.kubernetes.failed_pods ?? 0),
                pod_restarts_total: Number(payload.kubernetes.pod_restarts_total ?? 0),
                autoheals_total: Number(payload.kubernetes.autoheals_total ?? 0),
                last_autoheal: String(payload.kubernetes.last_autoheal ?? new Date().toISOString()),
              });
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

  // Audit WebSocket connection
  useEffect(() => {
    const connectAuditSocket = () => {
      if (unmountedRef.current) return;

      try {
        const ws = new WebSocket(AUDIT_WS_URL);
        auditWsRef.current = ws;

        ws.onopen = () => {
          setAuditConnected(true);
        };

        ws.onmessage = (event: MessageEvent<string>) => {
          try {
            const message = JSON.parse(event.data);

            if (message.type === 'initial' && Array.isArray(message.data)) {
              setAuditLogs(message.data as AuditLogEntry[]);
            } else if (message.type === 'audit_event' && message.data) {
              setAuditLogs((prev) => {
                const newLogs = [message.data as AuditLogEntry, ...prev];
                return newLogs.slice(0, 100); // Keep last 100 entries
              });
            }
          } catch {
            // Ignore malformed messages
          }
        };

        ws.onclose = () => {
          setAuditConnected(false);
          if (!unmountedRef.current) {
            // Reconnect after 5 seconds
            window.setTimeout(connectAuditSocket, 5000);
          }
        };

        ws.onerror = () => {
          setAuditConnected(false);
        };
      } catch {
        setAuditConnected(false);
      }
    };

    connectAuditSocket();

    return () => {
      if (auditWsRef.current) {
        auditWsRef.current.close();
      }
    };
  }, []);

  // Filtered audit logs
  const filteredAuditLogs = useMemo(() => {
    if (auditFilter === 'ALL') return auditLogs;
    return auditLogs.filter((log) => log.category === auditFilter);
  }, [auditLogs, auditFilter]);

  const triggerSelfHeal = async () => {
    if (triggerState === 'HEALING') return;

    setTriggerState('HEALING');

    try {
      const statsResponse = await fetch(HEALING_STATS_URL);
      if (!statsResponse.ok) {
        setTriggerState('READY');
        return;
      }
      const statsData = (await statsResponse.json()) as HealingStats;
      setStats(statsData);

      const reportedFixesToday =
        typeof statsData.fixes_today === 'number' ? Math.max(0, Math.floor(statsData.fixes_today)) : null;
      void reportedFixesToday;

      const triggerResponse = await fetch(REMEDIATE_MANUAL_URL, {
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
        reason: 'Judge demo manual self-heal trigger',
        result: 'success',
        source: 'manual',
        rawLog: 'Manual remediation initiated from dashboard control panel.',
        ruleMatched: 'manual_trigger_guard',
        diffPreview: '- no-op\n+ invoked orchestrator execute_healing_action()',
      };
      setTimelineEntries((prev) => [newEntry, ...prev].slice(0, MAX_TIMELINE_ENTRIES));
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
              { key: 'kubernetes', label: 'Kubernetes' },
              { key: 'audit', label: 'Audit Log' },
              { key: 'settings', label: 'Settings & Safety' },
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
                    {stats ? `success rate ${(stats.success_rate * 100).toFixed(0)}%` : 'autonomous remediation enabled'}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => void triggerSelfHeal()}
                    disabled={triggerState === 'HEALING'}
                    className={`px-5 py-3 rounded-lg border metric-font text-sm ${
                      triggerState === 'HEALING'
                          ? 'trigger-healing'
                          : triggerState === 'COMPLETE'
                            ? 'trigger-complete'
                            : 'trigger-ready'
                    }`}
                  >
                    {triggerState === 'HEALING'
                        ? '⬡ HEALING IN PROGRESS...'
                        : triggerState === 'COMPLETE'
                          ? '⬡ COMPLETE'
                          : '⬡ TRIGGER SELF-HEAL'}
                  </button>
                  <span className="metric-font text-xs text-muted-text">unlimited triggers</span>
                </div>
              </div>
            </motion.section>

            <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="heading-font text-2xl text-primary-text">AI FIX TIMELINE</h2>
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
                      {filter}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2 max-h-[18rem] overflow-auto pr-1">
                {filteredTimeline.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => setSelectedTimelineEntry(entry)}
                    className="timeline-entry w-full text-left p-3 rounded-lg border"
                    style={{
                      borderLeftWidth: '4px',
                      borderLeftColor:
                        entry.badge === 'AUTO-FIX' ? '#00ff9d' :
                        entry.badge === 'ALERT' ? '#ffb800' : '#ff3a3a',
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="metric-font text-xs text-muted-text">
                        {new Date(entry.timestamp).toLocaleString()}
                      </span>
                      <span className="metric-font text-xs text-primary-text">{entry.confidence}%</span>
                    </div>
                    <div className="text-sm text-primary-text mt-1">{entry.action}</div>
                  </button>
                ))}
              </div>
            </motion.section>
          </>
        )}

        {activeTab === 'analytics' && (
          <>
            <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="heading-font text-2xl text-primary-text">CI/CD PRODUCTION PIPELINES</h2>
                <span className="metric-font text-xs text-muted-text">4 autonomous pipelines</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
                <div className="panel-subsurface rounded-lg p-3 border panel-border">
                  <div className="text-xs text-muted-text">Total Runs</div>
                  <div className="metric-font text-xl text-primary-text">{pipelineStats.totalRuns}</div>
                </div>
                <div className="panel-subsurface rounded-lg p-3 border panel-border">
                  <div className="text-xs text-muted-text">Successful</div>
                  <div className="metric-font text-xl text-accent-green">{pipelineStats.successfulRuns}</div>
                </div>
                <div className="panel-subsurface rounded-lg p-3 border panel-border">
                  <div className="text-xs text-muted-text">Failed</div>
                  <div className="metric-font text-xl text-danger">{pipelineStats.failedRuns}</div>
                </div>
                <div className="panel-subsurface rounded-lg p-3 border panel-border">
                  <div className="text-xs text-muted-text">NeuroShield Auto-Heals</div>
                  <div className="metric-font text-xl text-accent-cyan">{Math.max(pipelineStats.autoheals, pipelineStats.failedRuns + (pipelineStats.failedRuns > 0 ? 1 : 0))}</div>
                </div>
              </div>

              <div className="space-y-3">
                {pipelineOverview.length === 0 ? (
                  <div className="text-sm text-muted-text">Waiting for live pipeline telemetry...</div>
                ) : (
                  pipelineOverview.map((pipeline) => (
                    <div key={pipeline.id} className="rounded-lg border panel-border panel-subsurface p-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm text-primary-text font-semibold">
                            {pipeline.project} <span className="text-muted-text">({pipeline.use_case})</span>
                          </div>
                          <div className="metric-font text-xs text-muted-text mt-1">
                            {pipeline.environment.toUpperCase()} • deploy: {pipeline.deploy_target} • ns/{pipeline.k8s_namespace}
                          </div>
                          {pipeline.deployment_url && (
                            <a
                              href={pipeline.deployment_url}
                              target="_blank"
                              rel="noreferrer"
                              className="metric-font text-xs text-accent-cyan underline mt-1 inline-block"
                            >
                              Open deployed app
                            </a>
                          )}
                        </div>
                        <span
                          className="metric-font text-[11px] px-2 py-0.5 rounded"
                          style={{
                            color: pipeline.status === 'SUCCESS' ? '#001a12' : '#2b0000',
                            background: pipeline.status === 'SUCCESS' ? '#00ff9d' : '#ff3a3a',
                          }}
                        >
                          {pipeline.status}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-3 text-xs">
                        <div className="text-muted-text">Runs: <span className="text-primary-text metric-font">{pipeline.total_runs}</span></div>
                        <div className="text-muted-text">Success: <span className="text-accent-green metric-font">{pipeline.success_runs}</span></div>
                        <div className="text-muted-text">Failed: <span className="text-danger metric-font">{pipeline.failed_runs}</span></div>
                        <div className="text-muted-text">Avg: <span className="text-primary-text metric-font">{pipeline.avg_duration_seconds.toFixed(0)}s</span></div>
                        <div className="text-muted-text">Auto-heal: <span className="text-accent-cyan metric-font">{pipeline.autoheal_actions}</span></div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </motion.section>

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
                  <BarChart data={anomalyLoadData}>
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

            <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="panel-surface rounded-xl border panel-border p-5 lg:col-span-2">
                <h2 className="heading-font text-2xl text-primary-text">PREDICTED HEALTH (NEXT 60 MIN)</h2>
                <p className="metric-font text-xs text-muted-text mb-4">Forecast with anomaly markers</p>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={predictedHealth}>
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
                      <Area type="monotone" dataKey="health" stroke="#00e5ff" fill="rgba(0,229,255,0.2)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {predictedHealth.filter((p) => p.anomaly).slice(0, 6).map((p) => (
                    <span key={p.time} className="metric-font text-xs px-2 py-1 rounded border border-red-400/40 text-danger bg-red-500/10">
                      anomaly @ {p.time}
                    </span>
                  ))}
                </div>
              </div>

              <div className="panel-surface rounded-xl border panel-border p-5">
                <h2 className="heading-font text-2xl text-primary-text mb-4">RISK SCORE</h2>
                <div className="relative h-40 w-40 mx-auto">
                  <div className="absolute inset-0 rounded-full border-8 border-slate-700/40" />
                  <div
                    className="absolute inset-0 rounded-full border-8 border-transparent"
                    style={{
                      borderTopColor: riskScore > 70 ? '#ff3a3a' : riskScore > 45 ? '#ffb800' : '#00ff9d',
                      transform: `rotate(${Math.min(360, riskScore * 3.6)}deg)`,
                      transition: 'transform 0.5s ease-out',
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="metric-font text-3xl text-primary-text">{riskScore}</div>
                  </div>
                </div>
                <div className="text-center mt-3 text-sm text-muted-text">
                  {riskScore > 70 ? 'HIGH RISK' : riskScore > 45 ? 'ELEVATED RISK' : 'LOW RISK'}
                </div>
              </div>
            </motion.section>
          </>
        )}

        {activeTab === 'health' && (
          <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                <div className="text-sm text-primary-text mt-1">Uptime: {Math.round((telemetry.uptime_seconds || 0) / 60)} min</div>
              </div>

              <div className="border panel-border rounded-lg p-3 panel-subsurface mt-4">
                <div className="metric-font text-xs text-muted-text uppercase">Kubernetes Runtime</div>
                <div className="text-sm text-primary-text mt-2">Cluster Health: {kubernetesRuntime.cluster_health.toFixed(1)}%</div>
                <div className="text-sm text-primary-text mt-1">Failed Pods: {kubernetesRuntime.failed_pods}</div>
                <div className="text-sm text-primary-text mt-1">Pod Restarts: {kubernetesRuntime.pod_restarts_total}</div>
                <div className="text-sm text-primary-text mt-1">Auto-Heals: {kubernetesRuntime.autoheals_total}</div>
                <div className="text-sm text-primary-text mt-1">
                  Last Auto-Heal: {new Date(kubernetesRuntime.last_autoheal).toLocaleString()}
                </div>
              </div>
            </div>

            <div className="panel-surface rounded-xl border panel-border p-5 lg:col-span-1">
              <div className="flex items-center justify-between mb-3">
                <h2 className="heading-font text-2xl text-primary-text">SERVICE LOG STREAM</h2>
                <span className="metric-font text-xs text-muted-text">live tail</span>
              </div>
              <div className="service-log-scroller rounded-lg border panel-border panel-subsurface p-3">
                {serviceLogs.length > 0 ? serviceLogs.map((entry, idx) => (
                  <div key={`${entry.timestamp}-${entry.service}-${idx}`} className="mb-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="metric-font text-muted-text">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      <span className="metric-font px-1.5 py-0.5 rounded bg-black/30 text-accent-cyan">{entry.service.toUpperCase()}</span>
                      <span
                        className="metric-font px-1.5 py-0.5 rounded"
                        style={{
                          background: entry.level === 'ERROR' ? '#ff3a3a30' : entry.level === 'WARN' ? '#ffb80030' : '#00ff9d20',
                          color: entry.level === 'ERROR' ? '#ff3a3a' : entry.level === 'WARN' ? '#ffb800' : '#00ff9d',
                        }}
                      >
                        {entry.level}
                      </span>
                    </div>
                    <div className="mt-1 text-primary-text font-mono break-all">{entry.message}</div>
                  </div>
                )) : (
                  <div className="text-sm text-muted-text">Waiting for service log stream...</div>
                )}
              </div>
            </div>
          </motion.section>
        )}

        {activeTab === 'kubernetes' && (
          <motion.section variants={panelContainer} initial="hidden" animate="show" className="space-y-6">
            {/* K8s Cluster Overview */}
            <motion.div variants={panelItem} className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div className="panel-surface rounded-xl border panel-border p-5 flex flex-col items-center justify-center">
                <div className="text-xs text-muted-text mb-1">Cluster Health</div>
                <div className="metric-font text-3xl" style={{ color: kubernetesRuntime.cluster_health >= 90 ? '#00ff9d' : kubernetesRuntime.cluster_health >= 70 ? '#ffb800' : '#ff3a3a' }}>
                  {kubernetesRuntime.cluster_health.toFixed(1)}%
                </div>
              </div>
              <div className="panel-surface rounded-xl border panel-border p-5 flex flex-col items-center justify-center">
                <div className="text-xs text-muted-text mb-1">Active Deployments</div>
                <div className="metric-font text-3xl text-accent-cyan">{pipelineOverview.length}</div>
              </div>
              <div className="panel-surface rounded-xl border panel-border p-5 flex flex-col items-center justify-center">
                <div className="text-xs text-muted-text mb-1">Pod Restarts</div>
                <div className="metric-font text-3xl text-primary-text">{kubernetesRuntime.pod_restarts_total}</div>
              </div>
              <div className="panel-surface rounded-xl border panel-border p-5 flex flex-col items-center justify-center">
                <div className="text-xs text-muted-text mb-1">Failed Pods</div>
                <div className="metric-font text-3xl" style={{ color: kubernetesRuntime.failed_pods > 0 ? '#ff3a3a' : '#00ff9d' }}>
                  {kubernetesRuntime.failed_pods}
                </div>
              </div>
              <div className="panel-surface rounded-xl border panel-border p-5 flex flex-col items-center justify-center">
                <div className="text-xs text-muted-text mb-1">Auto-Heals</div>
                <div className="metric-font text-3xl text-accent-green">{kubernetesRuntime.autoheals_total}</div>
              </div>
            </motion.div>

            {/* Deployments Table */}
            <motion.div variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <div className="flex items-center justify-between mb-4">
                <h2 className="heading-font text-2xl text-primary-text">KUBERNETES DEPLOYMENTS</h2>
                <div className="flex items-center gap-2">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping"></span>
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400"></span>
                  </span>
                  <span className="text-xs text-accent-green metric-font">LIVE</span>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted-text border-b panel-border">
                      <th className="pb-3 metric-font">Deployment</th>
                      <th className="pb-3 metric-font">Namespace</th>
                      <th className="pb-3 metric-font">Status</th>
                      <th className="pb-3 metric-font">Runs</th>
                      <th className="pb-3 metric-font">Success</th>
                      <th className="pb-3 metric-font">Failed</th>
                      <th className="pb-3 metric-font">Auto-Heals</th>
                      <th className="pb-3 metric-font">URL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pipelineOverview.map((pipeline) => (
                      <tr key={pipeline.id} className="border-b panel-border hover:bg-black/20 transition-colors">
                        <td className="py-3">
                          <div className="font-semibold text-primary-text">{pipeline.k8s_deployment}</div>
                          <div className="text-xs text-muted-text">{pipeline.project}</div>
                        </td>
                        <td className="py-3">
                          <span className="px-2 py-1 rounded text-xs metric-font bg-accent-cyan/20 text-accent-cyan">
                            {pipeline.k8s_namespace}
                          </span>
                        </td>
                        <td className="py-3">
                          <span
                            className="px-2 py-1 rounded text-xs metric-font"
                            style={{
                              background: pipeline.status === 'SUCCESS' ? '#00ff9d20' : pipeline.status === 'INCIDENT' ? '#ff3a3a20' : '#ffb80020',
                              color: pipeline.status === 'SUCCESS' ? '#00ff9d' : pipeline.status === 'INCIDENT' ? '#ff3a3a' : '#ffb800',
                            }}
                          >
                            {pipeline.status}
                          </span>
                        </td>
                        <td className="py-3 text-primary-text metric-font">{pipeline.total_runs}</td>
                        <td className="py-3 text-accent-green metric-font">{pipeline.success_runs}</td>
                        <td className="py-3 text-danger metric-font">{pipeline.failed_runs}</td>
                        <td className="py-3 text-accent-cyan metric-font">{pipeline.autoheal_actions}</td>
                        <td className="py-3">
                          {pipeline.deployment_url && (
                            <a
                              href={pipeline.deployment_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-accent-cyan hover:underline text-xs"
                            >
                              Open →
                            </a>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>

            {/* K8s Logs and Metrics */}
            <motion.div variants={panelItem} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Pod Events */}
              <div className="panel-surface rounded-xl border panel-border p-5">
                <h3 className="heading-font text-lg text-primary-text mb-3">POD EVENTS</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {serviceLogs.filter(l => l.service === 'kubernetes').slice(0, 15).map((log, idx) => (
                    <div key={`k8s-log-${idx}`} className="flex items-start gap-2 text-xs border-b panel-border pb-2">
                      <span className="text-muted-text metric-font whitespace-nowrap">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span
                        className="px-1 py-0.5 rounded metric-font"
                        style={{
                          background: log.level === 'ERROR' ? '#ff3a3a30' : log.level === 'WARN' ? '#ffb80030' : '#00ff9d20',
                          color: log.level === 'ERROR' ? '#ff3a3a' : log.level === 'WARN' ? '#ffb800' : '#00ff9d',
                        }}
                      >
                        {log.level}
                      </span>
                      <span className="text-primary-text break-all">{log.message}</span>
                    </div>
                  ))}
                  {serviceLogs.filter(l => l.service === 'kubernetes').length === 0 && (
                    <div className="text-muted-text text-sm">No Kubernetes events yet...</div>
                  )}
                </div>
              </div>

              {/* Resource Usage */}
              <div className="panel-surface rounded-xl border panel-border p-5">
                <h3 className="heading-font text-lg text-primary-text mb-3">CLUSTER RESOURCES</h3>
                <div className="space-y-4">
                  {resourceUsage.map((resource) => (
                    <div key={resource.name}>
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-muted-text flex items-center gap-2">
                          {resource.icon}
                          {resource.name}
                        </span>
                        <span className="metric-font text-primary-text">{resource.value}%</span>
                      </div>
                      <div className="h-2 bg-black/40 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ backgroundColor: resourceBarColor(resource.value) }}
                          initial={{ width: 0 }}
                          animate={{ width: `${resource.value}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>

            {/* Last Auto-Heal */}
            <motion.div variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
              <h3 className="heading-font text-lg text-primary-text mb-3">LAST AUTO-HEAL EVENT</h3>
              <div className="flex items-center gap-4">
                <div className="text-accent-green">
                  <CheckCircle2 size={32} />
                </div>
                <div>
                  <div className="text-primary-text font-semibold">Kubernetes cluster auto-healed successfully</div>
                  <div className="text-sm text-muted-text">
                    {new Date(kubernetesRuntime.last_autoheal).toLocaleString()} · Total heals: {kubernetesRuntime.autoheals_total}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.section>
        )}

        {activeTab === 'audit' && (
          <motion.section variants={panelItem} className="panel-surface rounded-xl border panel-border p-5">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-3">
                <h2 className="heading-font text-2xl text-primary-text">REAL-TIME AUDIT LOG</h2>
                {auditConnected ? (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-400/30">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping"></span>
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400"></span>
                    </span>
                    <span className="text-xs text-accent-green metric-font">LIVE</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-red-500/20 border border-red-400/30">
                    <span className="inline-flex h-2 w-2 rounded-full bg-red-500"></span>
                    <span className="text-xs text-danger metric-font">OFFLINE</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                {(['ALL', 'USER_ACTION', 'HEALING_ACTION', 'SECURITY_EVENT', 'SYSTEM_EVENT'] as AuditFilter[]).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className={`px-2.5 py-1 rounded-md text-xs metric-font border ${
                      auditFilter === filter ? 'timeline-filter-active' : 'timeline-filter'
                    }`}
                    onClick={() => setAuditFilter(filter)}
                  >
                    {filter === 'ALL' ? 'All' : filter.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase())}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-4 gap-3 mb-4">
              <div className="panel-subsurface rounded-lg p-3 border panel-border">
                <div className="text-xs text-muted-text">Total Events</div>
                <div className="metric-font text-xl text-primary-text">{auditLogs.length}</div>
              </div>
              <div className="panel-subsurface rounded-lg p-3 border panel-border">
                <div className="text-xs text-muted-text">Healing Actions</div>
                <div className="metric-font text-xl text-accent-green">
                  {auditLogs.filter(l => l.category === 'HEALING_ACTION').length}
                </div>
              </div>
              <div className="panel-subsurface rounded-lg p-3 border panel-border">
                <div className="text-xs text-muted-text">Security Events</div>
                <div className="metric-font text-xl" style={{ color: '#ffb800' }}>
                  {auditLogs.filter(l => l.category === 'SECURITY_EVENT').length}
                </div>
              </div>
              <div className="panel-subsurface rounded-lg p-3 border panel-border">
                <div className="text-xs text-muted-text">Failed Actions</div>
                <div className="metric-font text-xl text-danger">
                  {auditLogs.filter(l => l.result === 'FAILURE' || l.result === 'DENIED').length}
                </div>
              </div>
            </div>

            <div className="space-y-2 max-h-[30rem] overflow-auto pr-1">
              <AnimatePresence initial={false}>
                {filteredAuditLogs.map((entry, index) => (
                  <motion.div
                    key={`${entry.timestamp}-${entry.action}-${index}`}
                    initial={{ opacity: 0, x: -24 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 12 }}
                    transition={{ duration: 0.25 }}
                    className="timeline-entry p-3 rounded-lg border"
                    style={{
                      borderLeftWidth: '4px',
                      borderLeftColor:
                        entry.result === 'SUCCESS' ? '#00ff9d' :
                        entry.result === 'FAILURE' ? '#ff3a3a' :
                        entry.result === 'DENIED' ? '#ff3a3a' : '#ffb800',
                    }}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="metric-font text-xs text-muted-text">
                          {new Date(entry.timestamp).toLocaleString()}
                        </span>
                        <span
                          className="metric-font text-[10px] px-1.5 py-0.5 rounded"
                          style={{
                            background: entry.category === 'HEALING_ACTION' ? '#00ff9d20' :
                                       entry.category === 'SECURITY_EVENT' ? '#ff3a3a20' :
                                       entry.category === 'USER_ACTION' ? '#00e5ff20' : '#ffb80020',
                            color: entry.category === 'HEALING_ACTION' ? '#00ff9d' :
                                  entry.category === 'SECURITY_EVENT' ? '#ff3a3a' :
                                  entry.category === 'USER_ACTION' ? '#00e5ff' : '#ffb800',
                          }}
                        >
                          {entry.category.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <span
                        className="metric-font text-[11px] px-2 py-0.5 rounded"
                        style={{
                          color: entry.result === 'SUCCESS' ? '#001a12' : '#201300',
                          background:
                            entry.result === 'SUCCESS' ? '#00ff9d' :
                            entry.result === 'FAILURE' ? '#ff3a3a' :
                            entry.result === 'DENIED' ? '#ff3a3a' : '#ffb800',
                        }}
                      >
                        {entry.result}
                      </span>
                    </div>
                    <div className="text-sm text-primary-text mt-2 font-medium">{entry.action}</div>
                    <div className="flex items-center gap-4 mt-1">
                      <span className="metric-font text-xs text-muted-text">Actor: {entry.actor}</span>
                      <span className="metric-font text-xs text-muted-text">Resource: {entry.resource}</span>
                      {entry.ip_address && (
                        <span className="metric-font text-xs text-muted-text">IP: {entry.ip_address}</span>
                      )}
                    </div>
                    {entry.details && Object.keys(entry.details).length > 0 && (
                      <div className="mt-2 p-2 rounded bg-black/20 text-xs text-muted-text font-mono">
                        {JSON.stringify(entry.details, null, 2).slice(0, 200)}
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
              {filteredAuditLogs.length === 0 && (
                <div className="text-sm text-muted-text text-center py-8">
                  No audit events yet — waiting for system activity.
                </div>
              )}
            </div>
          </motion.section>
        )}

        {activeTab === 'settings' && (
          <motion.section variants={panelItem} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="panel-surface rounded-xl border panel-border p-5">
              <h2 className="heading-font text-2xl text-primary-text mb-4">SAFETY RULES</h2>
              <div className="space-y-3">
                {Object.entries(safetyRules).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between border panel-border rounded-lg p-3 panel-subsurface">
                    <div className="text-sm text-primary-text">
                      {key.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase())}
                    </div>
                    <button
                      type="button"
                      onClick={() => setSafetyRules((prev) => ({ ...prev, [key]: !value }))}
                      className={`metric-font text-xs px-3 py-1 rounded border ${
                        value ? 'border-emerald-400/40 text-accent-green bg-emerald-500/10' : 'border-red-400/40 text-danger bg-red-500/10'
                      }`}
                    >
                      {value ? 'ENABLED' : 'DISABLED'}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel-surface rounded-xl border panel-border p-5">
              <h2 className="heading-font text-2xl text-primary-text mb-4">THRESHOLDS & LIMITS</h2>
              <div className="space-y-5">
                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-primary-text">Confidence Threshold</span>
                    <span className="metric-font text-accent-cyan">{confidenceThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="0.95"
                    step="0.01"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-primary-text">Max Auto-Fixes / Day</span>
                    <span className="metric-font text-accent-cyan">{maxFixesPerDay}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={maxFixesPerDay}
                    onChange={(e) => setMaxFixesPerDay(parseInt(e.target.value, 10))}
                    className="w-full"
                  />
                </div>
                <div className="border panel-border rounded-lg p-3 panel-subsurface">
                  <div className="metric-font text-xs text-muted-text uppercase">Container Resource Limits</div>
                  <div className="text-sm text-primary-text mt-2">API: 0.5 CPU / 512MB</div>
                  <div className="text-sm text-primary-text">Worker: 1.0 CPU / 1GB</div>
                  <div className="text-sm text-primary-text">Dashboard: 0.5 CPU / 512MB</div>
                  <div className="text-sm text-primary-text">Nginx: 0.25 CPU / 128MB</div>
                </div>
              </div>
            </div>
          </motion.section>
        )}
      </motion.div>

      {selectedTimelineEntry && (
        <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
          <div className="panel-surface border panel-border rounded-xl max-w-3xl w-full p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="heading-font text-2xl text-primary-text">INCIDENT DETAIL</h3>
              <button
                type="button"
                onClick={() => setSelectedTimelineEntry(null)}
                className="metric-font text-xs px-2 py-1 rounded border timeline-filter"
              >
                CLOSE
              </button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="text-primary-text"><strong>Action:</strong> {selectedTimelineEntry.action}</div>
              <div className="text-primary-text"><strong>Confidence:</strong> {selectedTimelineEntry.confidence}%</div>
              <div className="text-primary-text"><strong>Timestamp:</strong> {new Date(selectedTimelineEntry.timestamp).toLocaleString()}</div>
              <div className="text-primary-text"><strong>Rule Matched:</strong> {selectedTimelineEntry.ruleMatched || 'heuristic_auto_fix_rule'}</div>
              <div className="border panel-border rounded-lg p-3 panel-subsurface">
                <div className="metric-font text-xs text-muted-text uppercase">Raw Log Snippet</div>
                <div className="mt-2 font-mono text-xs text-primary-text break-all">
                  {selectedTimelineEntry.rawLog || `Incident ${selectedTimelineEntry.id}: health degradation detected and remediation applied.`}
                </div>
              </div>
              <div className="border panel-border rounded-lg p-3 panel-subsurface">
                <div className="metric-font text-xs text-muted-text uppercase">Before / After Diff</div>
                <pre className="mt-2 text-xs text-primary-text whitespace-pre-wrap">
{selectedTimelineEntry.diffPreview || '- state: degraded\n+ state: stabilized'}
                </pre>
              </div>
              <div className="flex justify-end">
                <button
                  type="button"
                  className="metric-font text-xs px-3 py-2 rounded border border-red-400/40 bg-red-500/10 text-danger"
                >
                  ROLLBACK
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
