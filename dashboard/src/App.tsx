import React, { useEffect, useState } from 'react';
import { ShieldAlert, Server, Activity, CheckCircle, Clock3, BarChart3, WifiOff } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';

interface Stats { active_heals: number; success_rate: number; avg_response_time: number; cost_saved: number; }
interface Action { action_name: string; duration_ms: number; success: boolean; }
interface Metric { time: string; success_rate: number; incidents: number; }

export default function App() {
  const [stats, setStats] = useState<Stats>({ active_heals: 0, success_rate: 0, avg_response_time: 0, cost_saved: 0 });
  const [actions, setActions] = useState<Action[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [connected, setConnected] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statsRes = await fetch('/api/dashboard/stats');
        if (statsRes.ok) {
          const s = await statsRes.json();
          setStats({
            active_heals: s.total_heals || 0,
            success_rate: s.success_rate || 0,
            avg_response_time: Math.round(s.avg_response_time || 0),
            cost_saved: s.cost_saved || 0
          });
        }

        const histRes = await fetch('/api/dashboard/history?limit=10');
        if (histRes.ok) {
          const h = await histRes.json();
          setActions(h.actions || []);
        }

        const metRes = await fetch('/api/dashboard/metrics');
        if (metRes.ok) {
          const m = await metRes.json();
          setMetrics(m.metrics || []);
        }
        setConnected(true);
      } catch (e) {
        setConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        
        <header className="flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
              <ShieldAlert className="text-blue-600" />
              NeuroShield Enterprise
            </h1>
            <p className="text-slate-500 text-sm mt-1">Live AI Orchestration & Platform Intelligence</p>
          </div>
          <div className="flex items-center gap-4">
            {!connected && (
              <span className="flex items-center gap-1 text-red-500 bg-red-50 px-3 py-1 rounded-full text-sm font-medium border border-red-100">
                <WifiOff size={16} /> Disconnected
              </span>
            )}
            <span className="flex items-center gap-2 text-blue-700 bg-blue-50 px-4 py-2 rounded-lg text-sm font-medium border border-blue-100">
              <Server size={18} /> API Connected
            </span>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { label: 'Active Resolutions', value: stats.active_heals, icon: <Activity size={24}/>, hint: 'Events today' },
            { label: 'Platform Reliability', value: parseFloat(stats.success_rate.toString()).toFixed(1) + '%', icon: <CheckCircle size={24}/>, hint: 'Success rate' },
            { label: 'MTTR Target', value: stats.avg_response_time + 'ms', icon: <Clock3 size={24}/>, hint: 'Mean time to resolve' },
            { label: 'Value Protected', value: '$' + Math.round(stats.cost_saved), icon: <BarChart3 size={24}/>, hint: 'Cost avoided' }
          ].map((card, i) => (
            <div key={i} className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between h-36 border-t-4 border-t-blue-500">
              <div className="flex justify-between items-start text-slate-500">
                <span className="text-xs font-bold uppercase tracking-wider">{card.label}</span>
                <span className="text-blue-500">{card.icon}</span>
              </div>
              <div>
                <div className="text-3xl font-extrabold text-slate-900">{card.value}</div>
                <div className="text-xs text-slate-400 font-medium mt-1">{card.hint}</div>
              </div>
            </div>
          ))}
        </section>

        <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">System Health Trends</h2>
          <div className="h-80 w-full">
            {metrics.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Area type="monotone" dataKey="success_rate" stroke="#2563eb" fill="#eff6ff" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
               <div className="flex items-center justify-center h-full text-slate-400 font-medium bg-slate-50 rounded-lg border border-dashed border-slate-300">
                 Awaiting telemetry payload...
               </div>
            )}
          </div>
        </section>

      </div>
    </div>
  );
}
