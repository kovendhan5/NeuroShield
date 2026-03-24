import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend
} from 'recharts';
import { TrendingUp, Activity, Zap, DollarSign, Server } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [kpis, setKpis] = useState({
    total_heals: 292,
    successful_heals: 205,
    failed_heals: 87,
    success_rate: 70.2,
    avg_confidence: 82.5,
    cost_saved: 10920,
    downtime_prevented: 450,
  });
  const [actions, setActions] = useState([
    { timestamp: '2026-03-24T10:54:00Z', action_name: 'restart_pod', success: true, duration_ms: 250 },
    { timestamp: '2026-03-24T10:53:00Z', action_name: 'scale_up', success: true, duration_ms: 340 },
    { timestamp: '2026-03-24T10:52:00Z', action_name: 'rollback_deploy', success: true, duration_ms: 520 },
  ]);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/dashboard', { timeout: 5000 });
        setKpis(response.data.kpis);
        setActions(response.data.recent_actions || []);
      } catch (error) {
        console.log('Using mock data');
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const actionData = [
    { name: 'restart_pod', value: 95, fill: '#00ff88' },
    { name: 'scale_up', value: 78, fill: '#00ccff' },
    { name: 'rollback_deploy', value: 56, fill: '#ff006e' },
    { name: 'retry_build', value: 42, fill: '#ffd60a' },
  ];

  const trendData = [
    { time: '00:00', confidence: 65, success: 72 },
    { time: '06:00', confidence: 72, success: 75 },
    { time: '12:00', confidence: 78, success: 80 },
    { time: '18:00', confidence: 85, success: 88 },
    { time: '23:59', confidence: 82, success: 85 },
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to br, #0f0f1e 0%, #1a1a2e 50%, #0f0f1e 100%)' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid rgba(139, 92, 246, 0.2)',
        backdropFilter: 'blur(20px)',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ maxWidth: '80rem', margin: '0 auto', padding: '16px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              background: 'linear-gradient(to r, #00ff88, #00ccff)',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 'bold',
              color: '#000'
            }}>N</div>
            <h1 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              background: 'linear-gradient(to r, #00ff88, #00ccff, #00ccff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>NeuroShield Executive</h1>
          </div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <button
              onClick={() => setActiveTab('overview')}
              style={{
                padding: '8px 24px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 'bold',
                background: activeTab === 'overview' ? 'linear-gradient(to r, #00ff88, #00ccff)' : 'transparent',
                color: activeTab === 'overview' ? '#000' : '#ccc',
                transition: 'all 0.3s ease'
              }}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              style={{
                padding: '8px 24px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 'bold',
                background: activeTab === 'analytics' ? 'linear-gradient(to r, #00ccff, #0099ff)' : 'transparent',
                color: activeTab === 'analytics' ? '#000' : '#ccc',
                transition: 'all 0.3s ease'
              }}
            >
              Analytics
            </button>
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '80rem', margin: '0 auto', padding: '32px 24px' }}>
        {activeTab === 'overview' && (
          <>
            {/* KPI Cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '24px',
              marginBottom: '32px'
            }}>
              <KPICard title="Total Healing" value="292" suffix="" color="#00ff88" />
              <KPICard title="Success Rate" value="70.2" suffix="%" color="#00ccff" />
              <KPICard title="Cost Saved" value="$10,920" suffix="" color="#ff006e" />
              <KPICard title="ML Confidence" value="82.5" suffix="%" color="#ffd60a" />
            </div>

            {/* Charts */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px', marginBottom: '32px' }}>
              {/* Pie Chart */}
              <div style={{
                background: 'rgba(147, 112, 219, 0.1)',
                backdropFilter: 'blur(20px)',
                borderRadius: '24px',
                padding: '32px',
                border: '1px solid rgba(0, 255, 136, 0.2)',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
              }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', color: '#00ff88' }}>Action Breakdown</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie data={actionData} cx="50%" cy="50%" outerRadius={80} dataKey="value">
                      {actionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} opacity={0.8} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '2px solid #00ff88', borderRadius: '12px', color: '#fff' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Line Chart */}
              <div style={{
                background: 'rgba(147, 112, 219, 0.1)',
                backdropFilter: 'blur(20px)',
                borderRadius: '24px',
                padding: '32px',
                border: '1px solid rgba(0, 204, 255, 0.2)',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
              }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', color: '#00ccff' }}>Performance Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(100, 100, 150, 0.2)" />
                    <XAxis dataKey="time" stroke="#8899aa" />
                    <YAxis stroke="#8899aa" />
                    <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '2px solid #00ccff', borderRadius: '12px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="confidence" stroke="#00ccff" strokeWidth={3} dot={{ fill: '#00ccff', r: 5 }} />
                    <Line type="monotone" dataKey="success" stroke="#00ff88" strokeWidth={3} dot={{ fill: '#00ff88', r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Actions */}
            <div style={{
              background: 'rgba(147, 112, 219, 0.1)',
              backdropFilter: 'blur(20px)',
              borderRadius: '24px',
              padding: '32px',
              border: '1px solid rgba(139, 92, 246, 0.2)',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', color: '#ffd60a' }}>Recent Healing Actions</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(139, 92, 246, 0.3)' }}>
                      <th style={{ textAlign: 'left', padding: '12px', color: '#00ccff', fontWeight: 'bold' }}>Timestamp</th>
                      <th style={{ textAlign: 'left', padding: '12px', color: '#00ccff', fontWeight: 'bold' }}>Action</th>
                      <th style={{ textAlign: 'left', padding: '12px', color: '#00ccff', fontWeight: 'bold' }}>Duration</th>
                      <th style={{ textAlign: 'left', padding: '12px', color: '#00ccff', fontWeight: 'bold' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {actions.slice(0, 8).map((action, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(139, 92, 246, 0.1)', hover: 'bg(rgba(139, 92, 246, 0.1))' }}>
                        <td style={{ padding: '12px', color: '#ccc', fontSize: '14px' }}>{new Date(action.timestamp).toLocaleTimeString()}</td>
                        <td style={{ padding: '12px' }}>
                          <span style={{ padding: '6px 12px', borderRadius: '9999px', fontSize: '12px', fontWeight: 'bold', background: 'rgba(139, 92, 246, 0.3)', color: '#00ccff', border: '1px solid rgba(0, 204, 255, 0.3)' }}>
                            {action.action_name}
                          </span>
                        </td>
                        <td style={{ padding: '12px', color: '#ccc', fontSize: '14px', fontFamily: 'monospace' }}>{action.duration_ms}ms</td>
                        <td style={{ padding: '12px' }}>
                          <span style={{
                            padding: '6px 12px',
                            borderRadius: '9999px',
                            fontSize: '12px',
                            fontWeight: 'bold',
                            background: action.success ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 0, 110, 0.2)',
                            color: action.success ? '#00ff88' : '#ff006e',
                            border: action.success ? '1px solid rgba(0, 255, 136, 0.3)' : '1px solid rgba(255, 0, 110, 0.3)'
                          }}>
                            {action.success ? '✓ Success' : '✗ Failed'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {activeTab === 'analytics' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
            {/* Business Impact */}
            <div style={{
              background: 'rgba(147, 112, 219, 0.1)',
              backdropFilter: 'blur(20px)',
              borderRadius: '24px',
              padding: '32px',
              border: '1px solid rgba(255, 0, 110, 0.2)',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', background: 'linear-gradient(to r, #ff006e, #ff4d6d)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Business Impact</h3>
              <div style={{ space: '16px' }}>
                <Metric label="Recovery Time MTTR" value="52s" subtext="vs 30min manual" />
                <Metric label="Cost per Incident" value="$5" subtext="vs $70 manual" />
                <Metric label="Annual Projection" value="$50K+" subtext="in savings" />
                <Metric label="Downtime Prevented" value="450h" subtext="cumulative" />
              </div>
            </div>

            {/* System Health */}
            <div style={{
              background: 'rgba(147, 112, 219, 0.1)',
              backdropFilter: 'blur(20px)',
              borderRadius: '24px',
              padding: '32px',
              border: '1px solid rgba(0, 204, 255, 0.2)',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '24px', background: 'linear-gradient(to r, #00ccff, #0099ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>System Health</h3>
              <div style={{ space: '12px' }}>
                <HealthIndicator name="API" status="healthy" latency="2ms" />
                <HealthIndicator name="PostgreSQL" status="healthy" latency="5ms" />
                <HealthIndicator name="Redis" status="healthy" latency="1ms" />
                <HealthIndicator name="Jenkins" status="healthy" latency="45ms" />
                <HealthIndicator name="Prometheus" status="healthy" latency="12ms" />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const KPICard = ({ title, value, suffix, color }) => (
  <div style={{
    background: `rgba(${color === '#00ff88' ? '0, 255, 136' : color === '#00ccff' ? '0, 204, 255' : color === '#ff006e' ? '255, 0, 110' : '255, 214, 10'}, 0.1)`,
    backdropFilter: 'blur(20px)',
    borderRadius: '24px',
    padding: '32px',
    border: `1px solid rgba(${color === '#00ff88' ? '0, 255, 136' : color === '#00ccff' ? '0, 204, 255' : color === '#ff006e' ? '255, 0, 110' : '255, 214, 10'}, 0.2)`,
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.3s ease',
    cursor: 'pointer',
  }}>
    <h4 style={{ fontSize: '12px', fontWeight: 'bold', color: '#999', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '24px' }}>{title}</h4>
    <p style={{ fontSize: '36px', fontWeight: 'bold', color: color }}>{value}<span style={{ fontSize: '16px', color: '#999', marginLeft: '8px' }}>{suffix}</span></p>
  </div>
);

const Metric = ({ label, value, subtext }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '16px', border: '1px solid rgba(139, 92, 246, 0.2)', marginBottom: '12px' }}>
    <span style={{ color: '#ccc', fontWeight: '600' }}>{label}</span>
    <div style={{ textAlign: 'right' }}>
      <p style={{ fontSize: '24px', fontWeight: 'bold', background: 'linear-gradient(to r, #00ff88, #00ccff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{value}</p>
      <p style={{ fontSize: '12px', color: '#777' }}>{subtext}</p>
    </div>
  </div>
);

const HealthIndicator = ({ name, status, latency }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '16px', border: '1px solid rgba(139, 92, 246, 0.2)', marginBottom: '12px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
      <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'linear-gradient(to r, #00ff88, #00ccff)', animation: 'pulse 2s ease-in-out infinite' }} />
      <span style={{ fontSize: '14px', fontWeight: '600', color: '#ddd' }}>{name}</span>
    </div>
    <span style={{ fontSize: '12px', color: '#777', fontFamily: 'monospace' }}>{latency}</span>
  </div>
);

export default Dashboard;
