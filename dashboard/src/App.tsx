import { useState, useEffect } from 'react';
import { Activity, AlertCircle, CheckCircle, Clock, TrendingUp, Zap, Download, Moon, Sun, Settings, Bell, BarChart3, PieChart as PieIcon, TrendingDown } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ScatterChart, Scatter } from 'recharts';

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

const colors = {
  bg: '#0f1117',
  card: '#161b22',
  border: '#30363d',
  text: '#c9d1d9',
  textDim: '#6e7681',
  blue: '#0969da',
  green: '#238636',
  red: '#da3633',
  amber: '#d29922',
  purple: '#8957e5',
  cyan: '#79c0ff',
};

function App() {
  const [isDark, setIsDark] = useState(true);
  const [timeRange, setTimeRange] = useState('24h');
  const [alert, setAlert] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  const [stats, setStats] = useState<DashboardStats>({
    active_heals: 3,
    total_success: 205,
    total_failed: 87,
    avg_response_time: 52,
    success_rate: 70.2,
    p95_response_time: 142,
    p99_response_time: 245,
    cost_saved: 10920,
    mttr_reduction: 97,
  });

  const [actions, setActions] = useState<HealingAction[]>([
    { timestamp: '2026-03-24T11:54:00Z', action_name: 'restart_pod', success: true, duration_ms: 52, confidence: 0.95, pod_name: 'incident-board-3a7f' },
    { timestamp: '2026-03-24T11:53:15Z', action_name: 'scale_up', success: true, duration_ms: 87, confidence: 0.88, pod_name: 'api-service-2d4b' },
    { timestamp: '2026-03-24T11:52:45Z', action_name: 'rollback_deploy', success: true, duration_ms: 120, confidence: 0.92, pod_name: 'frontend-5x2k' },
    { timestamp: '2026-03-24T11:51:30Z', action_name: 'retry_build', success: false, duration_ms: 45, confidence: 0.65, pod_name: 'build-service' },
    { timestamp: '2026-03-24T11:50:15Z', action_name: 'scale_up', success: true, duration_ms: 95, confidence: 0.81, pod_name: 'cache-service' },
  ]);

  const [metrics] = useState([
    { time: '00:00', success_rate: 65, confidence: 72, incidents: 12, mttr: 45, cost: 840 },
    { time: '04:00', success_rate: 72, confidence: 75, incidents: 10, mttr: 38, cost: 700 },
    { time: '08:00', success_rate: 78, confidence: 80, incidents: 8, mttr: 32, cost: 560 },
    { time: '12:00', success_rate: 85, confidence: 85, incidents: 5, mttr: 22, cost: 350 },
    { time: '16:00', success_rate: 88, confidence: 87, incidents: 3, mttr: 18, cost: 210 },
    { time: '20:00', success_rate: 91, confidence: 90, incidents: 2, mttr: 12, cost: 140 },
  ]);

  const [actionBreakdown] = useState([
    { name: 'restart_pod', value: 95, fill: colors.red },
    { name: 'scale_up', value: 78, fill: colors.blue },
    { name: 'rollback_deploy', value: 56, fill: colors.purple },
    { name: 'retry_build', value: 42, fill: colors.amber },
    { name: 'clear_cache', value: 21, fill: colors.cyan },
  ]);

  const [topIncidents] = useState([
    { incident: 'Pod OOM Kill', count: 45, severity: 'critical', trend: '-15%' },
    { incident: 'High CPU Usage', count: 38, severity: 'warning', trend: '-8%' },
    { incident: 'Deployment Failure', count: 32, severity: 'warning', trend: '-22%' },
    { incident: 'API Timeout', count: 28, severity: 'warning', trend: '+5%' },
    { incident: 'Memory Leak', count: 19, severity: 'info', trend: '-11%' },
  ]);

  const [services] = useState<Service[]>([
    { name: 'API Server', status: 'healthy', latency: 2, uptime: 99.98 },
    { name: 'PostgreSQL', status: 'healthy', latency: 5, uptime: 99.95 },
    { name: 'Redis Cache', status: 'healthy', latency: 1, uptime: 100 },
    { name: 'Jenkins', status: 'healthy', latency: 45, uptime: 99.92 },
    { name: 'Prometheus', status: 'warning', latency: 120, uptime: 99.5 },
    { name: 'Grafana', status: 'healthy', latency: 15, uptime: 99.99 },
  ]);

  const [responseTimeDistribution] = useState([
    { range: '<10ms', count: 120 },
    { range: '10-50ms', count: 450 },
    { range: '50-100ms', count: 320 },
    { range: '100-200ms', count: 95 },
    { range: '200ms+', count: 15 },
  ]);

  useEffect(() => {
    // Simulate critical alert
    if (Math.random() > 0.8) {
      setAlert('⚠️ High error rate detected (3.2%) - Automatic recovery initiated');
      setTimeout(() => setAlert(null), 5000);
    }
  }, []);

  const exportData = () => {
    const data = JSON.stringify({ stats, actions, metrics }, null, 2);
    const link = document.createElement('a');
    link.href = 'data:text/json,' + encodeURIComponent(data);
    link.download = `neuroshield-report-${new Date().toISOString()}.json`;
    link.click();
  };

  const bgColor = isDark ? colors.bg : '#ffffff';
  const cardColor = isDark ? colors.card : '#f6f8fa';
  const textColor = isDark ? colors.text : '#0d1117';
  const borderColor = isDark ? colors.border : '#e5e7eb';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: bgColor, color: textColor, transition: 'all 0.3s' }}>
      {/* Alert Banner */}
      {alert && (
        <div style={{ backgroundColor: colors.red, color: 'white', padding: '12px 24px', textAlign: 'center', fontSize: '14px', fontWeight: 'bold' }}>
          {alert}
        </div>
      )}

      {/* Header */}
      <header style={{ borderBottom: `1px solid ${borderColor}`, backgroundColor: cardColor, backdropFilter: 'blur(10px)' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: `linear-gradient(135deg, ${colors.blue} 0%, ${colors.green} 100%)`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Zap size={24} color="white" />
            </div>
            <div>
              <h1 style={{ fontSize: '20px', fontWeight: 'bold', margin: 0, color: textColor }}>NeuroShield</h1>
              <p style={{ fontSize: '12px', margin: '0', color: colors.textDim }}>Enterprise AIOps Platform</p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Time Range Selector */}
            <select value={timeRange} onChange={(e) => setTimeRange(e.target.value)} style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: cardColor, color: textColor, border: `1px solid ${borderColor}`, fontSize: '12px', cursor: 'pointer' }}>
              <option value="1h">Last 1 Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>

            {/* Export Button */}
            <button onClick={exportData} style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: colors.blue, color: 'white', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Download size={16} /> Export
            </button>

            {/* Notifications */}
            <button style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: cardColor, border: `1px solid ${borderColor}`, color: textColor, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Bell size={16} />
              <span style={{ fontSize: '12px', fontWeight: 'bold', color: colors.red }}>3</span>
            </button>

            {/* Theme Toggle */}
            <button onClick={() => setIsDark(!isDark)} style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: cardColor, border: `1px solid ${borderColor}`, color: textColor, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>

            {/* Settings */}
            <button style={{ padding: '8px 12px', borderRadius: '6px', backgroundColor: cardColor, border: `1px solid ${borderColor}`, color: textColor, cursor: 'pointer' }}>
              <Settings size={16} />
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div style={{ borderBottom: `1px solid ${borderColor}`, backgroundColor: cardColor, display: 'flex', maxWidth: '1400px', margin: '0 auto' }}>
        {['overview', 'analytics', 'health', 'incidents'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '12px 24px',
              border: 'none',
              backgroundColor: activeTab === tab ? colors.blue : 'transparent',
              color: activeTab === tab ? 'white' : textColor,
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: activeTab === tab ? 'bold' : 'normal',
              textTransform: 'capitalize',
              transition: 'all 0.3s',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '32px 24px' }}>
        {activeTab === 'overview' && (
          <>
            {/* KPI Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
              <StatCard icon={<Activity size={20} />} label="Active Heals" value={`${stats.active_heals}`} trend="Real-time" color={colors.blue} />
              <StatCard icon={<CheckCircle size={20} />} label="Success Rate" value={`${stats.success_rate.toFixed(1)}%`} trend="Target: 90%+" color={colors.green} />
              <StatCard icon={<TrendingDown size={20} />} label="Avg MTTR" value={`${stats.avg_response_time}ms`} trend="vs 30min manual" color={colors.purple} />
              <StatCard icon={<BarChart3 size={20} />} label="Cost Saved" value={`$${stats.cost_saved}`} trend="This month" color={colors.amber} />
              <StatCard icon={<TrendingUp size={20} />} label="MTTR Reduction" value={`${stats.mttr_reduction}%`} trend="Improvement" color={colors.cyan} />
            </div>

            {/* Main Charts Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginBottom: '24px' }}>
              {/* Performance & Cost Trend */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Performance & Cost Trend</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={metrics} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                    <defs>
                      <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={colors.blue} stopOpacity={0.8} />
                        <stop offset="95%" stopColor={colors.blue} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={borderColor} />
                    <XAxis dataKey="time" stroke={colors.textDim} />
                    <YAxis stroke={colors.textDim} />
                    <Tooltip contentStyle={{ backgroundColor: cardColor, border: `1px solid ${borderColor}` }} />
                    <Area type="monotone" dataKey="cost" stroke={colors.blue} fillOpacity={1} fill="url(#colorCost)" name="Cost Saved ($)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Action Breakdown */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Action Breakdown</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={actionBreakdown} cx="50%" cy="50%" innerRadius={60} outerRadius={100} dataKey="value">
                      {actionBreakdown.map((entry, idx) => (
                        <Cell key={idx} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: cardColor, border: `1px solid ${borderColor}` }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Healing Pipeline & Response Time */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
              {/* Real-Time Pipeline */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, marginBottom: '20px', margin: 0 }}>
                  <span style={{ marginRight: '8px' }}>⚡</span> Real-Time Healing Pipeline
                </h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '400px', overflowY: 'auto' }}>
                  {actions.map((action, idx) => (
                    <HealingActionCard key={idx} action={action} index={idx} borderColor={borderColor} />
                  ))}
                </div>
              </div>

              {/* Response Time Distribution */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Response Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={responseTimeDistribution} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={borderColor} />
                    <XAxis type="number" stroke={colors.textDim} />
                    <YAxis dataKey="range" type="category" stroke={colors.textDim} width={60} />
                    <Bar dataKey="count" fill={colors.blue} radius={[0, 8, 8, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {activeTab === 'analytics' && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
              {/* Success Rate Trend */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Success Rate Trend</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke={borderColor} />
                    <XAxis dataKey="time" stroke={colors.textDim} />
                    <YAxis stroke={colors.textDim} />
                    <Tooltip contentStyle={{ backgroundColor: cardColor, border: `1px solid ${borderColor}` }} />
                    <Legend />
                    <Line type="monotone" dataKey="success_rate" stroke={colors.green} strokeWidth={2} name="Success Rate %" />
                    <Line type="monotone" dataKey="confidence" stroke={colors.blue} strokeWidth={2} name="ML Confidence %" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Incident Trend */}
              <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Incident Reduction</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke={borderColor} />
                    <XAxis dataKey="time" stroke={colors.textDim} />
                    <YAxis stroke={colors.textDim} />
                    <Tooltip contentStyle={{ backgroundColor: cardColor, border: `1px solid ${borderColor}` }} />
                    <Bar dataKey="incidents" fill={colors.red} radius={[8, 8, 0, 0]} name="Incidents" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Top Incidents */}
            <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, marginBottom: '20px', margin: 0 }}>Top Incidents</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: `2px solid ${borderColor}` }}>
                    <th style={{ textAlign: 'left', padding: '12px', color: colors.textDim, fontWeight: 'bold', fontSize: '12px' }}>Incident</th>
                    <th style={{ textAlign: 'center', padding: '12px', color: colors.textDim, fontWeight: 'bold', fontSize: '12px' }}>Count</th>
                    <th style={{ textAlign: 'center', padding: '12px', color: colors.textDim, fontWeight: 'bold', fontSize: '12px' }}>Severity</th>
                    <th style={{ textAlign: 'center', padding: '12px', color: colors.textDim, fontWeight: 'bold', fontSize: '12px' }}>Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {topIncidents.map((inc, idx) => (
                    <tr key={idx} style={{ borderBottom: `1px solid ${borderColor}` }}>
                      <td style={{ padding: '12px', color: textColor }}>{inc.incident}</td>
                      <td style={{ textAlign: 'center', padding: '12px', color: textColor, fontWeight: 'bold' }}>{inc.count}</td>
                      <td style={{ textAlign: 'center', padding: '12px' }}>
                        <span style={{ padding: '4px 8px', borderRadius: '4px', fontSize: '12px', backgroundColor: inc.severity === 'critical' ? colors.red : inc.severity === 'warning' ? colors.amber : colors.green, color: 'white', fontWeight: 'bold' }}>
                          {inc.severity}
                        </span>
                      </td>
                      <td style={{ textAlign: 'center', padding: '12px', color: inc.trend.includes('-') ? colors.green : colors.red, fontWeight: 'bold' }}>{inc.trend}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {activeTab === 'health' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
            {services.map((service, idx) => (
              <ServiceHealthCard key={idx} service={service} colors={colors} cardColor={cardColor} borderColor={borderColor} textColor={textColor} />
            ))}
          </div>
        )}

        {activeTab === 'incidents' && (
          <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '24px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: textColor, margin: '0 0 24px 0' }}>Incident Management</h2>
            <div style={{ minHeight: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: colors.textDim }}>
              <p>Advanced incident management coming soon...</p>
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes slideIn {
          0% { opacity: 0; transform: translateY(10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

const StatCard = ({ icon, label, value, trend, color }: any) => (
  <div style={{ backgroundColor: colors.card, border: `1px solid ${colors.border}`, borderRadius: '12px', padding: '16px', animation: 'slideIn 0.3s ease-out' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
      <p style={{ fontSize: '12px', fontWeight: '600', color: colors.textDim, margin: 0 }}>{label}</p>
      <div style={{ color, opacity: 0.8 }}>{icon}</div>
    </div>
    <p style={{ fontSize: '28px', fontWeight: 'bold', color: colors.text, margin: '8px 0 4px 0' }}>{value}</p>
    <p style={{ fontSize: '11px', color: colors.textDim, margin: 0 }}>{trend}</p>
  </div>
);

const HealingActionCard = ({ action, index, borderColor }: any) => {
  const actionColors: any = {
    restart_pod: { bg: '#3d1920', border: colors.red, color: '#f85149' },
    scale_up: { bg: '#0d1f27', border: colors.blue, color: '#79c0ff' },
    rollback_deploy: { bg: '#3d1f52', border: colors.purple, color: '#d2a8ff' },
    retry_build: { bg: '#3d2f1e', border: colors.amber, color: '#ffd700' },
  };

  const style = actionColors[action.action_name] || actionColors.restart_pod;

  return (
    <div style={{ padding: '12px', borderRadius: '8px', backgroundColor: style.bg, border: `1px solid ${style.border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between', animation: `slideIn 0.3s ease-out ${index * 50}ms backwards`, fontSize: '13px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
        <div style={{ color: style.color, fontWeight: 'bold' }}>{action.action_name.split('_').join(' ')}</div>
        <div style={{ color: colors.textDim, fontSize: '11px' }}>{action.pod_name}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', textAlign: 'right', whiteSpace: 'nowrap' }}>
        <div style={{ fontSize: '11px' }}>
          <div style={{ color: colors.textDim }}>Conf</div>
          <div style={{ fontWeight: 'bold', color: style.color }}>{((action.confidence || 0.8) * 100).toFixed(0)}%</div>
        </div>
        <div style={{ fontSize: '11px' }}>
          <div style={{ color: colors.textDim }}>Time</div>
          <div style={{ fontFamily: 'monospace', fontWeight: 'bold', color: colors.text }}>{action.duration_ms}ms</div>
        </div>
      </div>
    </div>
  );
};

const ServiceHealthCard = ({ service, colors, cardColor, borderColor, textColor }: any) => {
  const statusColors = {
    healthy: colors.green,
    warning: colors.amber,
    critical: colors.red,
  };

  return (
    <div style={{ backgroundColor: cardColor, border: `1px solid ${borderColor}`, borderRadius: '12px', padding: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: statusColors[service.status], animation: 'pulse 2s infinite' }}></div>
        <h3 style={{ fontSize: '16px', fontWeight: 'bold', color: textColor, margin: 0 }}>{service.name}</h3>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
        <div>
          <p style={{ fontSize: '11px', color: colors.textDim, margin: '0 0 4px 0' }}>Latency</p>
          <p style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: 0 }}>{service.latency}ms</p>
        </div>
        <div>
          <p style={{ fontSize: '11px', color: colors.textDim, margin: '0 0 4px 0' }}>Uptime</p>
          <p style={{ fontSize: '18px', fontWeight: 'bold', color: textColor, margin: 0 }}>{service.uptime}%</p>
        </div>
      </div>
      <div style={{ marginTop: '12px', height: '4px', backgroundColor: borderColor, borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ height: '100%', backgroundColor: statusColors[service.status], width: `${service.uptime}%` }}></div>
      </div>
    </div>
  );
};

export default App;
