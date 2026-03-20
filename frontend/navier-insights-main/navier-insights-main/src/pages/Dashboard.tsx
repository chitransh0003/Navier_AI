import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Zap, Activity, Clock, Database } from 'lucide-react';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
import { toast } from 'sonner';

const SEGMENTS = [
  { id: 'RKL-04', status: 'SAFE' as const, pressure: '45.0 bar', flow: '0.25 m³/s', scs: '96.8%', fluid: 'HSD', warningPos: null },
  { id: 'MUM-12', status: 'WARNING' as const, pressure: '42.0 bar', flow: '0.20 m³/s', scs: '71.3%', fluid: 'MS', warningPos: 40 },
  { id: 'CHN-07', status: 'SAFE' as const, pressure: '38.5 bar', flow: '0.22 m³/s', scs: '89.1%', fluid: 'ATF', warningPos: null },
];

const ANALYSES = [
  { time: '10:42 AM', segment: 'RKL-04', fluid: 'HSD', status: 'SAFE' as const, score: '0.103', highlight: false },
  { time: '10:38 AM', segment: 'MUM-12', fluid: 'MS', status: 'WARNING' as const, score: '0.219', highlight: false },
  { time: '10:31 AM', segment: 'CHN-07', fluid: 'ATF', status: 'SAFE' as const, score: '0.141', highlight: false },
  { time: '09:55 AM', segment: 'RKL-04', fluid: 'HSD', status: 'SAFE' as const, score: '0.098', highlight: false },
  { time: '09:22 AM', segment: 'MUM-12', fluid: 'MS', status: 'CRITICAL' as const, score: '0.289', highlight: true },
];

function SegmentPipe({ warningPos }: { warningPos: number | null }) {
  return (
    <svg className="w-full h-10" viewBox="0 0 300 40">
      <rect x="10" y="16" width="280" height="8" rx="4" fill="#E2E8F0" />
      <circle r="4" fill="#0A4FD4" opacity="0.8">
        <animateMotion dur="3s" repeatCount="indefinite" path="M10,20 L290,20" />
      </circle>
      {warningPos && (
        <g>
          <circle cx={warningPos * 2.8 + 10} cy="20" r="5" fill="#F59E0B" opacity="0.8">
            <animate attributeName="r" values="5;8;5" dur="1.5s" repeatCount="indefinite" />
          </circle>
          <circle cx={warningPos * 2.8 + 10} cy="20" r="10" fill="none" stroke="#F59E0B" strokeWidth="1.5" opacity="0.3">
            <animate attributeName="r" values="10;16;10" dur="1.5s" repeatCount="indefinite" />
          </circle>
        </g>
      )}
    </svg>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="pt-16">
        {/* Hero Banner */}
        <div className="bg-gradient-to-r from-primary/5 via-secondary/5 to-background px-8 py-10">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="font-heading text-3xl font-bold text-foreground mb-2">Pipeline Operations Center</h1>
              <p className="text-muted-foreground">Live AI monitoring · 3 active segments · Last updated: just now</p>
              <div className="flex flex-wrap gap-3 mt-4">
                {[
                  { text: '3 Segments Active', dot: 'bg-navier-safe' },
                  { text: '0 Active Alerts', dot: 'bg-navier-safe' },
                  { text: 'Avg SCS: 94.2%', dot: 'bg-primary' },
                  { text: 'Model: numpy backend', dot: 'bg-muted-foreground' },
                ].map((kpi, i) => (
                  <span key={i} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-card shadow-sm border border-border text-sm text-foreground">
                    <span className={`w-2 h-2 rounded-full ${kpi.dot}`} />
                    {kpi.text}
                  </span>
                ))}
              </div>
            </div>
            <svg width="120" height="60" viewBox="0 0 120 60" className="hidden md:block">
              <rect x="0" y="26" width="120" height="8" rx="4" fill="#E2E8F0" />
              <circle r="4" fill="#0A4FD4" opacity="0.7">
                <animateMotion dur="2.5s" repeatCount="indefinite" path="M0,30 L120,30" />
              </circle>
            </svg>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-8 py-8 space-y-8">
          {/* Segment Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            {SEGMENTS.map((seg, i) => (
              <motion.div
                key={seg.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="navier-card"
              >
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-heading font-bold text-foreground">Segment {seg.id}</h3>
                  <StatusBadge status={seg.status} />
                </div>
                <SegmentPipe warningPos={seg.warningPos} />
                <div className="grid grid-cols-3 gap-3 mt-3 text-center">
                  <div><p className="text-xs text-muted-foreground">Pressure</p><p className="text-sm font-semibold text-foreground">{seg.pressure}</p></div>
                  <div><p className="text-xs text-muted-foreground">Flow</p><p className="text-sm font-semibold text-foreground">{seg.flow}</p></div>
                  <div><p className="text-xs text-muted-foreground">SCS</p><p className="text-sm font-semibold text-foreground">{seg.scs}</p></div>
                </div>
                <button onClick={() => navigate('/simulate')} className="mt-4 text-sm font-medium text-primary hover:underline flex items-center gap-1">
                  Analyze <ArrowRight size={14} />
                </button>
              </motion.div>
            ))}
          </div>

          {/* Two column */}
          <div className="grid md:grid-cols-5 gap-6">
            {/* Recent Analyses */}
            <div className="md:col-span-3 navier-card">
              <h3 className="font-heading font-bold text-foreground mb-4">Recent AI Analyses</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border text-left text-muted-foreground">
                    <th className="pb-2 font-medium">Time</th><th className="pb-2 font-medium">Segment</th><th className="pb-2 font-medium">Fluid</th><th className="pb-2 font-medium">Status</th><th className="pb-2 font-medium">Score</th><th className="pb-2"></th>
                  </tr></thead>
                  <tbody>
                    {ANALYSES.map((a, i) => (
                      <tr key={i} className={`border-b border-border last:border-0 ${a.highlight ? 'bg-destructive/5' : i % 2 === 0 ? 'bg-background' : 'bg-muted/30'}`}>
                        <td className="py-2.5 text-muted-foreground">{a.time}</td>
                        <td className="py-2.5 font-medium text-foreground">{a.segment}</td>
                        <td className="py-2.5 text-foreground">{a.fluid}</td>
                        <td className="py-2.5"><StatusBadge status={a.status} /></td>
                        <td className="py-2.5 font-mono text-foreground">{a.score}</td>
                        <td className="py-2.5"><button className="text-primary text-xs font-medium hover:underline">View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* System Health */}
            <div className="md:col-span-2 space-y-6">
              <div className="navier-card">
                <h3 className="font-heading font-bold text-foreground mb-4">System Health</h3>
                <div className="space-y-3">
                  {[
                    { label: 'Model Backend', value: 'numpy (PINN-LSTM active)', dot: 'bg-navier-safe', icon: Activity },
                    { label: 'API Status', value: 'Demo mode · Offline', dot: 'bg-destructive', icon: Database },
                    { label: 'Avg Response', value: '18.4 ms', dot: 'bg-navier-warning', icon: Zap },
                    { label: 'Physics Engine', value: 'Navier-Stokes · Active', dot: 'bg-primary', icon: Activity },
                    { label: 'Last Dataset', value: '7,000 rows · Synthetic', dot: 'bg-muted-foreground', icon: Clock },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-3 text-sm">
                      <span className={`w-2 h-2 rounded-full ${item.dot} shrink-0`} />
                      <span className="text-muted-foreground w-28 shrink-0">{item.label}</span>
                      <span className="text-foreground font-medium truncate">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="navier-card">
                <h3 className="font-heading font-bold text-foreground mb-4">Quick Actions</h3>
                <div className="space-y-3">
                  <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={() => navigate('/simulate')} className="w-full gradient-bg text-primary-foreground font-semibold rounded-xl py-3 text-sm">
                    Run New Simulation
                  </motion.button>
                  <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={() => toast.info('Data generation would run: python data/synthetic_gen.py')} className="w-full border border-primary text-primary font-semibold rounded-xl py-3 text-sm hover:bg-primary/5 transition-colors">
                    Generate Synthetic Data
                  </motion.button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
