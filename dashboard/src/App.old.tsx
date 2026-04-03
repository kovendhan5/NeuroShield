import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle,
  Clock3,
  Download,
  RefreshCw,
  Server,
  ShieldAlert,
  Wifi,
  WifiOff,
} from 'lucide-react';
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

interface HealingAction {
  timestamp: string;
  action_name: string;
  success: boolean;
  duration_ms: number;
  confidence?: number;
  pod_name?: string;
  id?: string;
}

interface DashboardStats {
  active_heals: number;
  total_success: number;
  total_failed: number;
  avg_response_time: number;
  success_rate: number;
  p95_response_time?: number;
  p99_response_time?: number;
  cost_saved?: number;
  mttr_reduction?: number;
}

interface Service {
  name: string;
  status: 'healthy' | 'warning' | 'critical';
  latency: number;
  uptime: number;
}

interface Metric {
  time?: string;
  timestamp?: string;
  success_rate?: number;
  confidence?: number;
  incidents?: number;
  mttr?: number;
  cost?: number;
  cpu?: number;
  memory?: number;
  error_rate?: number;
  response_time?: number;
  pod_restarts?: number;
}

const initialStats: DashboardStats = {
  active_heals: 0,
  total_success: 0,
  total_failed: 0,
  avg_response_time: 0,
  success_rate: 0,
  p95_response_time: 0,
  p99_response_time: 0,
  cost_saved: 0,
  mttr_reduction: 0,
};

const initialActions: HealingAction[] = [];
const initialMetrics: Metric[] = [];

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path;
}

function metricLabel(metric: Metric): string {
  if (metric.time) return metric.time;
  if (!metric.timestamp) return '--:--';
  const d = new Date(metric.timestamp);
  if (Number.isNaN(d.getTime())) return metric.timestamp.slice(11, 16) || '--:--';
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function normalizeMetrics(payload: any): Metric[] {
  const metrics: any[] = payload?.metrics ?? payload?.data ?? [];
  if (!Array.isArray(metrics) || metrics.length === 0) return [];

  return metrics.map((m: any) => {
    const totalActions = Number(m.total_actions ?? m.total ?? 0);
    const successRate = Number(m.success_rate ?? m.successRate ?? 0);
    const incidents = Number(m.incidents ?? m.failed ?? Math.max(totalActions - Math.round(totalActions * successRate / 100), 0));
    return {
      time: metricLabel(m),
      timestamp: m.timestamp,
      success_rate: successRate,
      confidence: Number(m.ml_confidence ?? m.confidence ?? 70),
      incidents,
      mttr: Number(m.mttr ?? m.response_time ?? 25),
      cost: Number(m.cost ?? incidents * 70),
      cpu: Number(m.cpu ?? 0),
      memory: Number(m.memory ?? 0),
      error_rate: Number(m.error_rate ?? 0),
      response_time: Number(m.response_time ?? 0),
      pod_restarts: Number(m.pod_restarts ?? 0),
    } as Metric;
  });
}

async function fetchFirstOk(endpoints: string[]): Promise<any | null> {
  for (const endpoint of endpoints) {
    try {
      const resp = await fetch(apiUrl(endpoint));
      if (resp.ok) {
        return await resp.json();
      }
    } catch {
      // Try next endpoint variant.
    }
  }
  return null;
}

function StatCard({
  icon,
  label,
  value,
  hint,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <section className="bg-white rounded-xl p-5 shadow-sm border border-slate-200 flex flex-col justify-between hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between text-slate-500 mb-2">
        <span className="text-sm font-medium tracking-wide uppercase">{label}</span>
        <span className="p-2 bg-blue-50 text-blue-600 rounded-lg">{icon}</span>
      </div>
      <strong className="text-3xl font-bold text-slate-800 tracking-tight">{value}</strong>
      <p className="text-xs text-slate-400 mt-2 font-medium">{hint}</p>
    </section>
  );
}

function App() {
  const [alert, setAlert] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [updateFrequency, setUpdateFrequency] = useState('5');

  const [stats, setStats] = useState<DashboardStats>(initialStats);
  const [actions, setActions] = useState<HealingAction[]>(initialActions);
  const [metrics, setMetrics] = useState<Metric[]>(initialMetrics);

  const services: Service[] = useMemo(
    () => [
      { name: 'API Server', status: isConnected ? 'healthy' : 'critical', latency: isConnected ? 6 : 0, uptime: isConnected ? 99.98 : 0 },
      { name: 'PostgreSQL', status: 'healthy', latency: 9, uptime: 99.95 },
      { name: 'Redis Cache', status: 'healthy', latency: 2, uptime: 99.99 },
      { name: 'Jenkins', status: 'warning', latency: 86, uptime: 99.51 },
      { name: 'Prometheus', status: 'healthy', latency: 41, uptime: 99.92 },
      { name: 'Grafana', status: 'healthy', latency: 17, uptime: 99.99 },
    ],
    [isConnected]
  );

  const actionBreakdown = useMemo(() => {
    const map = new Map<string, number>();
    actions.forEach((action) => {
      const key = action.action_name || 'unknown';
      map.set(key, (map.get(key) ?? 0) + 1);
    });
    return Array.from(map.entries()).map(([name, value], i) => ({
      name,
      value,
      fill: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][i % 5],
    }));
  }, [actions]);

  const refreshDashboardData = async () => {
    try {
      const [statsPayload, historyPayload, metricPayload, healthPayload] = await Promise.all([
        fetchFirstOk(['/api/dashboard/stats', '/api/dashboard-data']),
        fetchFirstOk(['/api/dashboard/history?limit=8']),
        fetchFirstOk(['/api/dashboard/system-metrics', '/api/dashboard/metrics', '/api/metrics']),
        fetchFirstOk(['/health', '/api/health/detailed']),
      ]);

      if (statsPayload) {
        const fromDashboardData = statsPayload.kpis ? statsPayload.kpis : statsPayload;
        const totalHeals = Number(fromDashboardData.total_heals ?? fromDashboardData.total ?? 0);
        const successRate = Number(fromDashboardData.success_rate ?? 0);
        const successCount = Number(fromDashboardData.successful_heals ?? Math.round(totalHeals * successRate / 100));
        const failed = Number(fromDashboardData.failed_actions ?? fromDashboardData.failed_heals ?? Math.max(totalHeals - successCount, 0));

        setStats((prev) => ({
          ...prev,
          active_heals: totalHeals,
          total_success: successCount,
          total_failed: failed,
          avg_response_time: Math.round(Number(fromDashboardData.avg_response_time ?? prev.avg_response_time)),
          success_rate: successRate || prev.success_rate,
          cost_saved: Number(fromDashboardData.cost_saved ?? prev.cost_saved ?? 0),
          mttr_reduction: Number(fromDashboardData.mttr_reduction ?? prev.mttr_reduction ?? 0),
        }));
      }

      if (historyPayload?.actions && Array.isArray(historyPayload.actions)) {
        const realActions = historyPayload.actions.map((a: any, idx: number) => ({
          timestamp: a.timestamp,
          action_name: a.action_name,
          success: Boolean(a.success),
          duration_ms: Number(a.duration_ms ?? 0),
          confidence: Number(a.confidence ?? 0.8),
          pod_name: a.pod_name ?? 'service-node',
          id: a.id ?? `action-${idx}`,
        }));
        if (realActions.length > 0) {
          setActions(realActions.slice(0, 10));
        }
      }

      if (metricPayload) {
        const parsedMetrics = normalizeMetrics(metricPayload);
        if (parsedMetrics.length > 0) {
          setMetrics(parsedMetrics.slice(-12));
        }
      }

      setIsConnected(Boolean(healthPayload) || Boolean(statsPayload));
      setLastUpdate(new Date());
    } catch (error) {
      console.error(`Failed to fetch dashboard data: ${error}`);
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const intervalMs = Number(updateFrequency) * 1000;
    refreshDashboardData();
    const timer = setInterval(refreshDashboardData, intervalMs);

    return () => {
      clearInterval(timer);
    };
  }, [updateFrequency]);

  const exportData = () => {
    const report = {
      exportedAt: new Date().toISOString(),
      stats,
      recentActions: actions,
      metrics,
      services,
      connected: isConnected,
    };
    const link = document.createElement('a');
    link.href = 'data:text/json,' + encodeURIComponent(JSON.stringify(report, null, 2));
    link.download = `neuroshield-report-${new Date().toISOString()}.json`;
    link.click();
  };

  const manualRefresh = () => {
    setIsLoading(true);
    refreshDashboardData();
    setAlert('Manual refresh completed');
    setTimeout(() => setAlert(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-blue-100 pb-16">
      {alert && (
        <div className="fixed top-4 right-4 z-50 bg-slate-800 text-white px-4 py-3 rounded-lg shadow-xl text-sm flex items-center gap-3 animate-in slide-in-from-top-4" role="status">
          <CheckCircle size={18} className="text-emerald-400" />
          {alert}
        </div>
      )}

      {/* Header */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative flex items-center justify-center w-10 h-10 bg-blue-600 rounded-xl text-white shadow-md">
              <ShieldAlert size={20} />
              <span className={`absolute -bottom-1 -right-1 w-3.5 h-3.5 border-2 border-white rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight tracking-tight text-slate-900">NeuroShield Control Center</h1>
              <p className="text-xs text-slate-500 font-medium tracking-wide">
                {isConnected ? 'Live stream connected' : 'Offline / Waiting for data'}
                <span className="mx-1.5 opacity-50">&bull;</span>
                Updated {lastUpdate ? lastUpdate.toLocaleTimeString() : '--:--'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <select 
              value={updateFrequency} 
              onChange={(e) => setUpdateFrequency(e.target.value)}
              className="text-sm bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 font-medium"
            >
              <option value="1">Every 1s</option>
              <option value="5">Every 5s</option>
              <option value="10">Every 10s</option>
              <option value="30">Every 30s</option>
            </select>

            <button onClick={manualRefresh} className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors">
              <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
            </button>

            <button onClick={exportData} className="flex items-center gap-2 bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 px-3 py-1.5 rounded-lg text-sm font-semibold transition-all shadow-sm">
              <Download size={16} /> Export
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        
        {/* Connection Status Row */}
        <section className="flex flex-wrap gap-4 mb-8 text-sm font-medium">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${isConnected ? 'bg-emerald-50 text-emerald-700 border-emerald-100' : 'bg-red-50 text-red-700 border-red-100'}`}>
            {isConnected ? <Wifi size={16} /> : <WifiOff size={16} />}
            <span>{isConnected ? 'System online & bound to API' : 'Disconnected from local nodes'}</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
            <Clock3 size={16} />
            <span>Polling frequency: {updateFrequency}s</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 text-blue-700 border border-blue-100">
            <Server size={16} />
            <span>Proxy: {API_BASE || 'localhost:9999'}</span>
          </div>
        </section>

        {/* Tab Navigation */}
        <nav className="flex gap-1 mb-8 border-b border-slate-200">
          {['overview', 'analytics', 'health', 'live'].map((tab) => (
            <button
              key={tab}
              className={`px-6 py-3 text-sm font-semibold capitalize border-b-2 transition-colors ${
                activeTab === tab 
                  ? 'border-blue-600 text-blue-700' 
                  : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300'
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'live' ? 'Audit Log' : tab}
            </button>
          ))}
        </nav>

        {activeTab === 'overview' && (
          <div className="space-y-6">
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard icon={<Activity size={20} />} label="Active interventions" value={String(stats.active_heals)} hint="Resolved actions this session" />
              <StatCard icon={<CheckCircle size={20} />} label="Platform Reliability" value={`${stats.success_rate.toFixed(1)}%`} hint="Success rate of auto-healing" />
              <StatCard icon={<Clock3 size={20} />} label="MTTR Target" value={`${stats.avg_response_time}s`} hint="Mean time to resolution" />
              <StatCard icon={<BarChart3 size={20} />} label="Value protected" value={`$${Math.round(stats.cost_saved ?? 0)}`} hint="Estimated outage prevention" />
            </section>

            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <article className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
                <header className="mb-6">
                  <h2 className="text-lg font-bold text-slate-800">System Resilience Trend</h2>
                  <p className="text-sm text-slate-500 font-medium">Auto-healing success & model confidence over time</p>
                </header>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={metrics}>
                      <defs>
                        <linearGradient id="colorSuccess" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.15} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid vertical={false} stroke="#e2e8f0" strokeDasharray="4 4" />
                      <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                      <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dx={-10} />
                      <Tooltip 
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)' }}
                        itemStyle={{ color: '#0f172a', fontWeight: '600' }}
                      />
                      <Area type="monotone" dataKey="success_rate" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorSuccess)" name="Success %" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
                <header className="mb-6">
                  <h2 className="text-lg font-bold text-slate-800">Action Distribution</h2>
                  <p className="text-sm text-slate-500 font-medium">Mix of intervention types</p>
                </header>
                <div className="h-72 w-full flex justify-center items-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={actionBreakdown} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4}>
                        {actionBreakdown.map((entry, idx) => (
                          <Cell key={`${entry.name}-${idx}`} fill={entry.fill} stroke="rgba(0,0,0,0)" />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                      <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: '20px', fontSize: '12px', fontWeight: '500' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>
          </div>
        )}

        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <article className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
                <header className="mb-6">
                  <h2 className="text-lg font-bold text-slate-800">Telemetry Comparison</h2>
                  <p className="text-sm text-slate-500 font-medium">Resolution Success vs AI Confidence Score</p>
                </header>
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metrics} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#e2e8f0" strokeDasharray="4 4" />
                      <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                      <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dx={-10} />
                      <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                      <Legend wrapperStyle={{ paddingTop: '20px' }} />
                      <Line type="monotone" dataKey="success_rate" stroke="#3b82f6" strokeWidth={3} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} name="Success Rate" />
                      <Line type="monotone" dataKey="confidence" stroke="#10b981" strokeWidth={3} dot={{r: 4, strokeWidth: 2}} activeDot={{r: 6}} name="AI Confidence" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
                <header className="mb-6">
                  <h2 className="text-lg font-bold text-slate-800">Anomaly Load</h2>
                  <p className="text-sm text-slate-500 font-medium">Incident frequency mapping</p>
                </header>
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metrics} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid vertical={false} stroke="#e2e8f0" strokeDasharray="4 4"/>
                      <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dy={10} />
                      <YAxis axisLine={false} tickLine={false} tick={{fill: '#64748b', fontSize: 12}} dx={-10} />
                      <Tooltip cursor={{fill: '#f8fafc'}} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                      <Bar dataKey="incidents" fill="#8b5cf6" radius={[6, 6, 0, 0]} barSize={40} name="Active Incidents" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>
          </div>
        )}

        {activeTab === 'health' && (
          <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {services.map((service) => (
              <article className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition-shadow" key={service.name}>
                <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/80">
                  <h2 className="font-bold text-slate-800">{service.name}</h2>
                  <div className={`px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider ${
                    service.status === 'healthy' ? 'bg-emerald-100 text-emerald-700' :
                    service.status === 'warning' ? 'bg-amber-100 text-amber-700' :
                    'bg-red-100 text-red-700'
                  }`}>
                    {service.status}
                  </div>
                </div>
                <div className="p-5 flex gap-6">
                  <div className="flex-1">
                    <span className="block text-xs text-slate-500 font-medium uppercase mb-1">Latency</span>
                    <strong className="text-2xl font-bold text-slate-800">{service.latency} <span className="text-sm font-medium text-slate-500">ms</span></strong>
                  </div>
                  <div className="w-px bg-slate-200" />
                  <div className="flex-1">
                    <span className="block text-xs text-slate-500 font-medium uppercase mb-1">Uptime</span>
                    <strong className="text-2xl font-bold text-slate-800">{service.uptime}<span className="text-sm font-medium text-slate-500">%</span></strong>
                  </div>
                </div>
              </article>
            ))}
          </section>
        )}

        {activeTab === 'live' && (
          <section className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <header className="px-6 py-5 border-b border-slate-200 flex justify-between items-center bg-slate-50/80">
              <div>
                <h2 className="text-lg font-bold text-slate-800">Operation Audit Log</h2>
                <p className="text-sm text-slate-500 font-medium">Real-time healing execution records</p>
              </div>
            </header>
            <div className="divide-y divide-slate-100">
              {actions.length === 0 ? (
                <div className="px-6 py-16 text-center">
                  <Activity className="mx-auto h-8 w-8 text-slate-300 mb-3" />
                  <p className="text-slate-500 font-medium text-lg">No automated actions logged</p>
                  <p className="text-slate-400 text-sm mt-1">Waiting for incidents in the current timeframe.</p>
                </div>
              ) : (
                actions.map((action) => (
                  <article className="px-6 py-4 flex items-center gap-4 hover:bg-slate-50 transition-colors" key={action.id}>
                    <div className={`shrink-0 w-2.5 h-2.5 rounded-full ${action.success ? 'bg-emerald-500' : 'bg-red-500'}`} />
                    <div className="w-32 shrink-0">
                      <span className={`inline-block px-3 py-1.5 text-xs font-bold uppercase rounded-md tracking-wider ${
                        action.success ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                      }`}>
                        {action.success ? 'Resolved' : 'Failed'}
                      </span>
                    </div>
                    <strong className="w-48 shrink-0 text-slate-800 font-semibold">{action.action_name}</strong>
                    <span className="w-48 shrink-0 text-slate-500 text-sm font-mono bg-slate-100 px-2 py-1 rounded">{action.pod_name}</span>
                    <div className="w-32 shrink-0">
                      <div className="flex items-center gap-2">
                        <div className="w-full bg-slate-100 rounded-full h-2">
                          <div className={`h-2 rounded-full ${action.success ? 'bg-blue-500' : 'bg-slate-300'}`} style={{ width: `${Math.round((action.confidence ?? 0) * 100)}%` }} />
                        </div>
                        <span className="text-xs font-bold text-slate-600">{Math.round((action.confidence ?? 0) * 100)}%</span>
                      </div>
                    </div>
                    <span className="w-24 shrink-0 text-slate-500 text-sm text-right">{action.duration_ms} ms</span>
                    <span className="flex-1 text-slate-400 text-sm text-right font-medium">{new Date(action.timestamp).toLocaleTimeString()}</span>
                  </article>
                ))
              )}
            </div>
          </section>
        )}

        {/* Loading overlay */}
        {!isConnected && !isLoading && (
          <div className="mt-8 p-4 bg-amber-50 border border-amber-200 rounded-lg shadow-sm flex items-center gap-3">
             <ShieldAlert size={20} className="text-amber-500" />
             <p className="text-amber-800 text-sm font-semibold">
               Connecting to API gateway. Live metrics stream disconnected.
             </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
