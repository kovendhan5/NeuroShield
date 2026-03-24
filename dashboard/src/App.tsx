import { useState, useEffect } from 'react';
import { Activity, AlertCircle, CheckCircle, Clock, TrendingUp, Server, Zap } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import './App.css';

interface HealingAction {
  timestamp: string;
  action_name: string;
  success: boolean;
  duration_ms: number;
  confidence?: number;
  pod_name?: string;
}

interface DashboardStats {
  active_heals: number;
  total_success: number;
  total_failed: number;
  avg_response_time: number;
  success_rate: number;
}

function App() {
  const [stats, setStats] = useState<DashboardStats>({
    active_heals: 0,
    total_success: 205,
    total_failed: 87,
    avg_response_time: 52,
    success_rate: 70.2,
  });

  const [actions, setActions] = useState<HealingAction[]>([
    { timestamp: '2026-03-24T11:54:00Z', action_name: 'restart_pod', success: true, duration_ms: 52, confidence: 0.95, pod_name: 'incident-board-3a7f' },
    { timestamp: '2026-03-24T11:53:15Z', action_name: 'scale_up', success: true, duration_ms: 87, confidence: 0.88, pod_name: 'api-service-2d4b' },
    { timestamp: '2026-03-24T11:52:45Z', action_name: 'rollback_deploy', success: true, duration_ms: 120, confidence: 0.92, pod_name: 'frontend-5x2k' },
  ]);

  const [metrics, setMetrics] = useState([
    { time: '00:00', success_rate: 65, confidence: 72, incidents: 12 },
    { time: '04:00', success_rate: 72, confidence: 75, incidents: 10 },
    { time: '08:00', success_rate: 78, confidence: 80, incidents: 8 },
    { time: '12:00', success_rate: 85, confidence: 85, incidents: 5 },
    { time: '16:00', success_rate: 88, confidence: 87, incidents: 3 },
    { time: '20:00', success_rate: 91, confidence: 90, incidents: 2 },
  ]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/dashboard');
        const data = await response.json();
        setStats(data.stats || stats);
        setActions(data.recent_actions || actions);
      } catch (err) {
        console.log('Using demo data');
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-dark-950">
      {/* Header */}
      <header className="border-b border-dark-700 bg-dark-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-primary to-blue-active flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">NeuroShield</h1>
                <p className="text-xs text-dark-400">AIOps Healing Pipeline</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-dark-300">Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <StatCard
            icon={<Activity className="w-5 h-5" />}
            label="Active Heals"
            value={stats.active_heals}
            trend="+2 this hour"
            color="blue"
          />
          <StatCard
            icon={<CheckCircle className="w-5 h-5" />}
            label="Successful"
            value={stats.total_success}
            trend="205 total"
            color="green"
          />
          <StatCard
            icon={<AlertCircle className="w-5 h-5" />}
            label="Failed"
            value={stats.total_failed}
            trend="87 total"
            color="red"
          />
          <StatCard
            icon={<Clock className="w-5 h-5" />}
            label="Avg Response"
            value={`${stats.avg_response_time}ms`}
            trend="vs 30min manual"
            color="purple"
          />
          <StatCard
            icon={<TrendingUp className="w-5 h-5" />}
            label="Success Rate"
            value={`${stats.success_rate.toFixed(1)}%`}
            trend="Target: 90%+"
            color="amber"
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Healing Pipeline */}
          <div className="lg:col-span-2 bg-dark-900 border border-dark-700 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-primary" />
              Real-Time Healing Pipeline
            </h2>

            <div className="space-y-3">
              {actions.map((action, idx) => (
                <HealingActionCard key={idx} action={action} index={idx} />
              ))}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-6">Incident Summary</h2>

            <div className="space-y-4">
              <div className="p-4 bg-dark-800 rounded border-l-4 border-blue-primary">
                <p className="text-dark-300 text-sm">Total Incidents</p>
                <p className="text-2xl font-bold text-white">292</p>
              </div>
              <div className="p-4 bg-dark-800 rounded border-l-4 border-green-500">
                <p className="text-dark-300 text-sm">Auto-Resolved</p>
                <p className="text-2xl font-bold text-green-400">205</p>
              </div>
              <div className="p-4 bg-dark-800 rounded border-l-4 border-red-500">
                <p className="text-dark-300 text-sm">Escalated</p>
                <p className="text-2xl font-bold text-red-400">87</p>
              </div>

              <div className="pt-4 border-t border-dark-700">
                <div className="text-sm text-dark-400 mb-3">Success Rate</div>
                <div className="h-2 bg-dark-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-primary to-green-500 transition-all duration-1000"
                    style={{ width: `${stats.success_rate}%` }}
                  ></div>
                </div>
                <p className="text-xs text-dark-400 mt-2">{stats.success_rate.toFixed(1)}% of 292</p>
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Metrics */}
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-6">Performance Trend</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={metrics} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1d23', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="success_rate"
                  stroke="#0969da"
                  strokeWidth={2}
                  dot={{ fill: '#0969da', r: 4 }}
                  activeDot={{ r: 6 }}
                  name="Success Rate (%)"
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#79c0ff"
                  strokeWidth={2}
                  dot={{ fill: '#79c0ff', r: 4 }}
                  activeDot={{ r: 6 }}
                  name="ML Confidence (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Incident Trend */}
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-6">
            <h2 className="text-lg font-semibold text-white mb-6">Incident Reduction</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={metrics} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1d23', border: '1px solid #333' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar
                  dataKey="incidents"
                  fill="#0969da"
                  radius={[8, 8, 0, 0]}
                  name="Active Incidents"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  trend: string;
  color: 'blue' | 'green' | 'red' | 'purple' | 'amber';
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, trend, color }) => {
  const colorClass = {
    blue: 'text-blue-primary',
    green: 'text-green-500',
    red: 'text-red-500',
    purple: 'text-purple-500',
    amber: 'text-amber-500',
  }[color];

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-lg p-4 hover:border-dark-600 transition-colors animate-slide-in">
      <div className="flex items-center justify-between mb-3">
        <p className="text-dark-400 text-sm font-medium">{label}</p>
        <div className={colorClass}>{icon}</div>
      </div>
      <p className="text-2xl font-bold text-white mb-1">{value}</p>
      <p className="text-xs text-dark-500">{trend}</p>
    </div>
  );
};

interface HealingActionCardProps {
  action: HealingAction;
  index: number;
}

const HealingActionCard: React.FC<HealingActionCardProps> = ({ action, index }) => {
  const actionColors: Record<string, { bg: string; text: string; border: string }> = {
    restart_pod: { bg: 'bg-red-950', text: 'text-red-400', border: 'border-red-700' },
    scale_up: { bg: 'bg-blue-950', text: 'text-blue-400', border: 'border-blue-700' },
    rollback_deploy: { bg: 'bg-purple-950', text: 'text-purple-400', border: 'border-purple-700' },
    retry_build: { bg: 'bg-amber-950', text: 'text-amber-400', border: 'border-amber-700' },
  };

  const colors = actionColors[action.action_name] || actionColors.restart_pod;

  return (
    <div className={`p-4 rounded-lg border ${colors.border} ${colors.bg} flex items-center justify-between animate-slide-in`} style={{ animationDelay: `${index * 100}ms` }}>
      <div className="flex items-center gap-4 flex-1">
        <div className={`w-10 h-10 rounded-lg ${colors.bg} border ${colors.border} flex items-center justify-center`}>
          {action.success ? (
            <CheckCircle className="w-6 h-6 text-green-400" />
          ) : (
            <AlertCircle className="w-6 h-6 text-red-400" />
          )}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <p className={`font-semibold ${colors.text}`}>{action.action_name}</p>
            <span className="text-xs text-dark-500">on {action.pod_name}</span>
          </div>
          <p className="text-sm text-dark-400">
            {new Date(action.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-6 text-right">
        <div>
          <p className="text-xs text-dark-400">Confidence</p>
          <p className="font-semibold text-white">{((action.confidence || 0.8) * 100).toFixed(0)}%</p>
        </div>
        <div>
          <p className="text-xs text-dark-400">Duration</p>
          <p className="font-mono text-white">{action.duration_ms}ms</p>
        </div>
      </div>
    </div>
  );
};

export default App;
