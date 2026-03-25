import { Activity, BarChart3, CheckCircle, Clock3, Server, ShieldAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

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

interface MetricPoint {
  time: string;
  health_score: number;
  cpu: number;
  memory: number;
}

const WS_URL = 'ws://localhost/ws/telemetry';
const INITIAL_BACKOFF_MS = 3000;
const MAX_BACKOFF_MS = 30000;

export default function App() {
  const [telemetry, setTelemetry] = useState<TelemetryMessage>({
    cpu: 0,
    memory: 0,
    health_score: 100,
    active_alerts: 0,
    recent_fix: null,
    timestamp: new Date().toISOString(),
  });
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [connected, setConnected] = useState(false);

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const backoffRef = useRef<number>(INITIAL_BACKOFF_MS);
  const closedByUnmountRef = useRef<boolean>(false);

  useEffect(() => {
    closedByUnmountRef.current = false;

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
      if (closedByUnmountRef.current) {
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
            setMetrics((prev) => {
              const point: MetricPoint = {
                time: new Date(payload.timestamp).toLocaleTimeString(),
                health_score: payload.health_score,
                cpu: payload.cpu,
                memory: payload.memory,
              };
              const next = [...prev, point];
              return next.slice(-30);
            });
          } catch {
            // Ignore malformed payloads and keep current state.
          }
        };

        ws.onclose = () => {
          setConnected(false);
          if (!closedByUnmountRef.current) {
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
      closedByUnmountRef.current = true;
      clearReconnectTimer();
      if (websocketRef.current !== null) {
        websocketRef.current.close();
      }
    };
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
            {connected ? (
              <span className="flex items-center gap-2 text-emerald-700 bg-emerald-50 px-4 py-2 rounded-lg text-sm font-bold border border-emerald-200">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                LIVE
              </span>
            ) : (
              <span className="flex items-center gap-2 text-red-700 bg-red-50 px-4 py-2 rounded-lg text-sm font-bold border border-red-200">
                <span className="inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                RECONNECTING...
              </span>
            )}
            <span className="flex items-center gap-2 text-blue-700 bg-blue-50 px-4 py-2 rounded-lg text-sm font-medium border border-blue-100">
              <Server size={18} /> WebSocket Feed
            </span>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { label: 'CPU', value: `${telemetry.cpu.toFixed(1)}%`, icon: <Activity size={24} />, hint: 'Current utilization' },
            { label: 'Memory', value: `${telemetry.memory.toFixed(1)}%`, icon: <BarChart3 size={24} />, hint: 'Current utilization' },
            { label: 'Health Score', value: `${telemetry.health_score.toFixed(1)}%`, icon: <CheckCircle size={24} />, hint: 'Computed reliability' },
            { label: 'Active Alerts', value: telemetry.active_alerts, icon: <Clock3 size={24} />, hint: 'Open incidents' },
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
                  <Area type="monotone" dataKey="health_score" stroke="#2563eb" fill="#eff6ff" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-400 font-medium bg-slate-50 rounded-lg border border-dashed border-slate-300">
                Awaiting websocket telemetry...
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
