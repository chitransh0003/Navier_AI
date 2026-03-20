import { motion } from 'framer-motion';
import { TrendingUp, ArrowUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, ReferenceLine } from 'recharts';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';

const anomalyData = Array.from({ length: 24 }, (_, i) => ({
  time: `${String(i).padStart(2, '0')}:00`,
  score: 0.1 + Math.sin(i * 0.5) * 0.15 + Math.random() * 0.1 + (i === 14 ? 0.4 : 0) + (i === 20 ? 0.25 : 0),
}));

const classificationData = [
  { name: 'SAFE', value: 74, color: 'hsl(160, 84%, 39%)' },
  { name: 'WARNING', value: 21, color: 'hsl(38, 92%, 50%)' },
  { name: 'CRITICAL', value: 5, color: 'hsl(0, 84%, 60%)' },
];

const segmentTable = [
  { id: 'RKL-04', fluid: 'HSD', pressure: '44.8 bar', scs: '96.2%', analyses: 312, alerts: 0, status: 'SAFE' as const },
  { id: 'MUM-12', fluid: 'MS', pressure: '41.5 bar', scs: '72.1%', analyses: 289, alerts: 7, status: 'WARNING' as const },
  { id: 'CHN-07', fluid: 'ATF', pressure: '38.9 bar', scs: '88.7%', analyses: 246, alerts: 1, status: 'SAFE' as const },
  { id: 'DEL-01', fluid: 'BENZENE', pressure: '46.2 bar', scs: '93.4%', analyses: 198, alerts: 0, status: 'SAFE' as const },
  { id: 'BLR-09', fluid: 'LPG', pressure: '35.1 bar', scs: '67.8%', analyses: 156, alerts: 12, status: 'CRITICAL' as const },
];

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="pt-16">
        <div className="bg-gradient-to-r from-primary/5 to-secondary/5 px-8 py-8">
          <div className="max-w-7xl mx-auto">
            <h1 className="font-heading text-3xl font-bold text-foreground">Pipeline Analytics</h1>
            <p className="text-muted-foreground mt-1">Historical performance, model accuracy, and system health trends.</p>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-8 py-8 space-y-8">
          {/* Charts Row */}
          <div className="grid md:grid-cols-2 gap-6">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="navier-card">
              <h3 className="font-heading font-bold text-foreground mb-4">Anomaly Score Over Time</h3>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={anomalyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <Tooltip />
                  <ReferenceLine y={0.4} stroke="#F59E0B" strokeDasharray="5 5" label={{ value: 'Warning', fill: '#F59E0B', fontSize: 10 }} />
                  <ReferenceLine y={0.85} stroke="#EF4444" strokeDasharray="5 5" label={{ value: 'Critical', fill: '#EF4444', fontSize: 10 }} />
                  <Line type="monotone" dataKey="score" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} viewport={{ once: true }} className="navier-card">
              <h3 className="font-heading font-bold text-foreground mb-4">Classification Distribution</h3>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={classificationData} cx="50%" cy="45%" innerRadius={60} outerRadius={90} dataKey="value" label={({ name, value }) => `${name} ${value}%`}>
                    {classificationData.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Pie>
                  <Legend />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* Stats Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            {[
              { label: 'Total Analyses Today', value: '847', sub: '+12% from yesterday', icon: TrendingUp },
              { label: 'Average SCS Score', value: '91.4%', sub: null, icon: null },
              { label: 'NPW Localizations', value: '23', sub: 'leak events localized today', icon: null },
            ].map((s, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} viewport={{ once: true }} className="navier-card text-center">
                <p className="text-sm text-muted-foreground mb-2">{s.label}</p>
                <p className="font-heading text-4xl font-bold gradient-text">{s.value}</p>
                {s.sub && (
                  <p className="text-xs text-navier-safe mt-2 flex items-center justify-center gap-1">
                    <ArrowUp size={12} /> {s.sub}
                  </p>
                )}
                {s.label === 'Average SCS Score' && (
                  <div className="w-full h-2 rounded-full bg-muted mt-3 overflow-hidden">
                    <motion.div initial={{ width: 0 }} whileInView={{ width: '91.4%' }} transition={{ duration: 1 }} viewport={{ once: true }} className="h-full rounded-full gradient-bg" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>

          {/* Performance Table */}
          <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="navier-card">
            <h3 className="font-heading font-bold text-foreground mb-4">Segment Performance Summary</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-3 font-medium">Segment ID</th><th className="pb-3 font-medium">Fluid</th><th className="pb-3 font-medium">Avg Pressure</th>
                  <th className="pb-3 font-medium">Avg SCS</th><th className="pb-3 font-medium">Analyses</th><th className="pb-3 font-medium">Alerts</th><th className="pb-3 font-medium">Status</th>
                </tr></thead>
                <tbody>
                  {segmentTable.map((row, i) => (
                    <tr key={row.id} className={`border-b border-border last:border-0 ${i % 2 === 0 ? 'bg-background' : 'bg-muted/30'}`}>
                      <td className="py-3 font-medium text-foreground">{row.id}</td>
                      <td className="py-3 text-foreground">{row.fluid}</td>
                      <td className="py-3 text-foreground">{row.pressure}</td>
                      <td className="py-3 text-foreground">{row.scs}</td>
                      <td className="py-3 text-foreground">{row.analyses}</td>
                      <td className="py-3 text-foreground">{row.alerts}</td>
                      <td className="py-3"><StatusBadge status={row.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
