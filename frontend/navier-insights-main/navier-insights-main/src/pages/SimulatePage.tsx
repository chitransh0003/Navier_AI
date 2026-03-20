import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2 } from 'lucide-react';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
import { useNavierAnalysis, type AnalysisResult } from '../hooks/useNavierAnalysis';
import { CHEMICAL_PROFILES, FLUID_TYPES, type FluidType } from '../constants/chemicalProfiles';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const DEFAULT_PARAMS = {
  inlet_pressure: 4500000, outlet_pressure: 4020000, flow_rate: 0.25, temperature: 42,
  density: 840, viscosity: 0.0035, acoustic_vibration: 10.5, pipe_diameter: 0.508,
  wall_thickness: 0.0095, segment_length: 10000,
};

const PRESETS = {
  normal: { ...DEFAULT_PARAMS },
  leak: { ...DEFAULT_PARAMS, outlet_pressure: 2400000, flow_rate: 0.20, acoustic_vibration: 35.0 },
  critical: { ...DEFAULT_PARAMS, inlet_pressure: 3600000, outlet_pressure: 2100000, flow_rate: 0.14, acoustic_vibration: 68.0, density: 745, viscosity: 0.00055 },
};

const FIELDS: { key: keyof typeof DEFAULT_PARAMS; label: string; unit: string }[] = [
  { key: 'inlet_pressure', label: 'Inlet Pressure', unit: 'Pa' },
  { key: 'outlet_pressure', label: 'Outlet Pressure', unit: 'Pa' },
  { key: 'flow_rate', label: 'Flow Rate', unit: 'm³/s' },
  { key: 'temperature', label: 'Temperature', unit: '°C' },
  { key: 'density', label: 'Density', unit: 'kg/m³' },
  { key: 'viscosity', label: 'Viscosity', unit: 'Pa·s' },
  { key: 'acoustic_vibration', label: 'Acoustic Vibration', unit: 'units' },
  { key: 'pipe_diameter', label: 'Pipe Diameter', unit: 'm' },
  { key: 'wall_thickness', label: 'Wall Thickness', unit: 'm' },
  { key: 'segment_length', label: 'Segment Length', unit: 'm' },
];

function AnomalyGauge({ score }: { score: number }) {
  const angle = score * 180;
  const r = 60;
  const cx = 70, cy = 70;
  const endX = cx + r * Math.cos(Math.PI - (angle * Math.PI) / 180);
  const endY = cy - r * Math.sin(Math.PI - (angle * Math.PI) / 180);
  const large = angle > 90 ? 1 : 0;
  const color = score < 0.4 ? 'hsl(var(--navier-safe))' : score < 0.85 ? 'hsl(var(--navier-warning))' : 'hsl(var(--navier-critical))';
  return (
    <svg viewBox="0 0 140 85" className="w-40 mx-auto">
      <path d={`M10,70 A60,60 0 0,1 130,70`} fill="none" stroke="#E2E8F0" strokeWidth="10" strokeLinecap="round" />
      <motion.path
        d={`M10,70 A60,60 0 ${large},1 ${endX},${endY}`}
        fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }}
      />
      <text x="70" y="68" textAnchor="middle" className="font-heading" fontSize="18" fontWeight="700" fill="currentColor">
        {(score * 100).toFixed(1)}%
      </text>
    </svg>
  );
}

function PipelineVisualizer({ data }: { data: AnalysisResult['npw_localization'] & { segment_length: number } }) {
  const pipeWidth = 800;
  const leakX = (data.leak_distance_m / data.segment_length) * pipeWidth + 50;
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="navier-card mt-6">
      <h3 className="font-heading font-bold text-foreground mb-4">Leak Localization</h3>
      <svg viewBox="0 0 900 120" className="w-full">
        {/* Pipe */}
        <rect x="50" y="45" width={pipeWidth} height="12" rx="6" fill="#E2E8F0" />
        {/* Sensors */}
        <circle cx="50" cy="51" r="8" fill="hsl(var(--primary))" />
        <text x="50" y="80" textAnchor="middle" fontSize="10" fill="hsl(var(--navier-muted))">Upstream</text>
        <circle cx={pipeWidth + 50} cy="51" r="8" fill="hsl(var(--primary))" />
        <text x={pipeWidth + 50} y="80" textAnchor="middle" fontSize="10" fill="hsl(var(--navier-muted))">Downstream</text>
        {/* Leak */}
        <circle cx={leakX} cy="51" r="6" fill="hsl(var(--navier-critical))" />
        <circle cx={leakX} cy="51" r="14" fill="none" stroke="hsl(var(--navier-critical))" strokeWidth="2" opacity="0.4" className="animate-ping-leak" />
        <text x={leakX} y="30" textAnchor="middle" fontSize="11" fontWeight="600" fill="hsl(var(--navier-critical))">{data.leak_distance_m}m</text>
        {/* Flow arrow */}
        <polygon points={`${pipeWidth / 2 + 50},42 ${pipeWidth / 2 + 60},48 ${pipeWidth / 2 + 50},54`} fill="hsl(var(--primary))" opacity="0.5" />
        {/* Confidence bar */}
        <rect x="50" y="95" width={pipeWidth} height="6" rx="3" fill="#F1F5F9" />
        <rect x="50" y="95" width={pipeWidth * data.confidence} height="6" rx="3" fill="hsl(var(--navier-safe))" opacity="0.7" />
        <text x={pipeWidth + 60} y="101" fontSize="9" fill="hsl(var(--navier-muted))">{(data.confidence * 100).toFixed(0)}%</text>
      </svg>
      <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
        <span className="px-2 py-1 rounded bg-muted border border-border">Wave Speed: {data.wave_speed_ms} m/s</span>
        <span className="px-2 py-1 rounded bg-muted border border-border">Δt: {data.delta_t_s}s</span>
        <span className="px-2 py-1 rounded bg-muted border border-border">Method: {data.method}</span>
      </div>
    </motion.div>
  );
}

function ResultsPanel({ result, isDemoMode }: { result: AnalysisResult; isDemoMode: boolean }) {
  const probData = [
    { name: 'SAFE', value: result.class_probabilities.SAFE, color: 'hsl(var(--navier-safe))' },
    { name: 'WARNING', value: result.class_probabilities.WARNING, color: 'hsl(var(--navier-warning))' },
    { name: 'CRITICAL', value: result.class_probabilities.CRITICAL, color: 'hsl(var(--navier-critical))' },
  ];
  const flagColors: Record<string, string> = {
    HIGH_PINN_DIVERGENCE: 'status-critical',
    SENSOR_DRIFT_RISK_HIGH_TEMP: 'status-warning',
    ACOUSTIC_SPIKE_DETECTED: 'status-warning',
  };

  return (
    <motion.div initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} className="space-y-5">
      {isDemoMode && (
        <div className="px-4 py-2 rounded-xl bg-primary/10 border border-primary/20 text-primary text-sm font-medium">
          Demo Mode — using simulated data
        </div>
      )}
      {/* Status Banner */}
      <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}
        className={`p-4 rounded-2xl text-center font-heading font-bold text-lg ${
          result.classification === 'SAFE' ? 'bg-navier-safe/10 text-navier-safe' :
          result.classification === 'WARNING' ? 'bg-navier-warning/10 text-navier-warning' :
          'bg-navier-critical/10 text-navier-critical'
        }`}>
        {result.classification}
        <p className="text-sm font-normal mt-1 opacity-80">{result.alert_message}</p>
      </motion.div>

      <AnomalyGauge score={result.anomaly_score} />

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'SCS Score', value: `${result.scs_report.scs}%` },
          { label: 'Reynolds', value: result.physics_report.reynolds_number.toLocaleString() },
          { label: 'RUL', value: `${result.rul_days} days` },
          { label: 'PINN Loss', value: result.physics_report.pinn_loss_weight.toFixed(3) },
        ].map((s, i) => (
          <div key={i} className="p-3 rounded-xl bg-muted/50 border border-border text-center">
            <p className="text-xs text-muted-foreground">{s.label}</p>
            <p className="font-heading font-bold text-foreground mt-1">{s.value}</p>
          </div>
        ))}
      </div>

      {/* Flags */}
      {result.scs_report.flags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {result.scs_report.flags.map(f => (
            <span key={f} className={`text-xs px-2.5 py-1 rounded-full border font-medium ${flagColors[f] || 'bg-muted text-foreground border-border'}`}>
              {f.split('_').join(' ')}
            </span>
          ))}
        </div>
      )}

      {/* Class Probabilities */}
      <div className="navier-card">
        <h4 className="font-heading font-semibold text-foreground mb-3 text-sm">Class Probabilities</h4>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={probData} layout="vertical">
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis type="category" dataKey="name" width={70} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
            <Bar dataKey="value" radius={[0, 6, 6, 0]}>
              {probData.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-muted-foreground text-center">Processed in {result.processing_time_ms}ms · {result.model_backend} backend</p>
    </motion.div>
  );
}

export default function SimulatePage() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [fluid, setFluid] = useState<FluidType>('HSD');
  const [ambientTemp, setAmbientTemp] = useState(35);
  const { result, isLoading, isDemoMode, runAnalysis, reset } = useNavierAnalysis();

  const profile = CHEMICAL_PROFILES[fluid];

  const handleRun = () => {
    runAnalysis({ ...params, fluid_type: fluid, ambient_temp_c: ambientTemp });
  };

  const applyPreset = (preset: keyof typeof PRESETS) => {
    setParams(PRESETS[preset]);
    if (preset === 'critical') setFluid('MS');
    else setFluid('HSD');
    reset();
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <div className="pt-16">
        {/* Header */}
        <div className="bg-gradient-to-r from-primary/5 to-secondary/5 px-8 py-8">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div>
              <h1 className="font-heading text-3xl font-bold text-foreground">Pipeline Simulation</h1>
              <p className="text-muted-foreground mt-1">Configure sensor parameters and run the NAVIER AI model</p>
            </div>
            <div className="hidden md:block navier-card !p-3">
              <div className="flex items-center gap-2 text-sm">
                <span className="w-2 h-2 rounded-full bg-navier-safe" />
                <span className="font-medium text-foreground">PINN-LSTM Ready</span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">numpy backend · no GPU required</p>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-8 py-8">
          <div className="grid md:grid-cols-2 gap-8">
            {/* Left: Form */}
            <div className="space-y-6">
              <div className="navier-card">
                <div className="flex items-center justify-between mb-5">
                  <h3 className="font-heading font-bold text-foreground">Sensor Parameters</h3>
                  <span className="text-xs px-2 py-1 rounded-full bg-muted border border-border text-muted-foreground">10 params</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  {FIELDS.map(f => (
                    <div key={f.key}>
                      <label className="text-xs text-muted-foreground font-medium">{f.label}</label>
                      <div className="relative mt-1">
                        <input
                          type="number"
                          value={params[f.key]}
                          onChange={e => setParams(p => ({ ...p, [f.key]: parseFloat(e.target.value) || 0 }))}
                          className="w-full px-3 py-2 pr-12 rounded-lg border border-input bg-background text-foreground text-sm focus:ring-2 focus:ring-ring focus:outline-none"
                        />
                        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">{f.unit}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Fluid selector */}
                <div className="mt-5">
                  <label className="text-xs text-muted-foreground font-medium">Fluid Type</label>
                  <div className="flex gap-2 mt-2">
                    {FLUID_TYPES.map(ft => (
                      <button
                        key={ft}
                        onClick={() => setFluid(ft)}
                        className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                          fluid === ft ? 'gradient-bg text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        {ft}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Ambient temp */}
                <div className="mt-5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-muted-foreground font-medium">Ambient Temperature: {ambientTemp}°C</label>
                    {ambientTemp >= 42 && <span className="text-xs px-2 py-0.5 rounded-full status-warning border font-medium">Heat Drift Risk</span>}
                  </div>
                  <input type="range" min="20" max="55" value={ambientTemp} onChange={e => setAmbientTemp(Number(e.target.value))} className="w-full mt-2 accent-primary" />
                </div>

                {/* Presets */}
                <div className="mt-5">
                  <label className="text-xs text-muted-foreground font-medium mb-2 block">Scenario Presets</label>
                  <div className="flex gap-2">
                    {([['normal', 'Normal Flow', 'bg-navier-safe'], ['leak', 'Leak Scenario', 'bg-navier-warning'], ['critical', 'Critical Leak', 'bg-navier-critical']] as const).map(([key, label, dot]) => (
                      <button key={key} onClick={() => applyPreset(key)} className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border text-sm text-foreground hover:bg-muted transition-colors">
                        <span className={`w-2 h-2 rounded-full ${dot}`} />
                        {label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="mt-6 space-y-3">
                  <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={handleRun} disabled={isLoading}
                    className="w-full gradient-bg text-primary-foreground font-bold rounded-xl py-3 text-sm flex items-center justify-center gap-2 disabled:opacity-70">
                    {isLoading ? <><Loader2 className="animate-spin" size={16} /> Analyzing...</> : 'Run AI Analysis'}
                  </motion.button>
                  <button onClick={() => { setParams(DEFAULT_PARAMS); setFluid('HSD'); setAmbientTemp(35); reset(); }}
                    className="w-full border border-border text-foreground font-medium rounded-xl py-3 text-sm hover:bg-muted transition-colors">
                    Reset to Defaults
                  </button>
                </div>
              </div>

              {/* Chemical Profile */}
              <div className="navier-card">
                <h3 className="font-heading font-bold text-foreground mb-3">Chemical Profile — {fluid}</h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div><span className="text-muted-foreground">Density:</span> <span className="font-medium text-foreground">{profile.density} kg/m³</span></div>
                  <div><span className="text-muted-foreground">Wave Speed:</span> <span className="font-medium text-foreground">{profile.waveSpeed} m/s</span></div>
                  <div><span className="text-muted-foreground">Bulk Modulus:</span> <span className="font-medium text-foreground">{profile.bulkModulus}</span></div>
                  <div><span className="text-muted-foreground">Viscosity:</span> <span className="font-medium text-foreground">{profile.viscosityRef}</span></div>
                </div>
              </div>
            </div>

            {/* Right: Results */}
            <div>
              <AnimatePresence mode="wait">
                {result ? (
                  <ResultsPanel key="results" result={result} isDemoMode={isDemoMode} />
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="h-full flex items-center justify-center">
                    <div className="text-center p-10 border-2 border-dashed border-border rounded-2xl">
                      <svg viewBox="0 0 200 60" className="w-48 mx-auto mb-4 opacity-30">
                        <rect x="10" y="24" width="180" height="12" rx="6" fill="#CBD5E1" />
                        <circle r="5" fill="#93C5FD" opacity="0.6">
                          <animateMotion dur="3s" repeatCount="indefinite" path="M10,30 L190,30" />
                        </circle>
                      </svg>
                      <p className="text-muted-foreground text-sm">Run simulation to see AI analysis results</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Pipeline Visualizer */}
          {result?.npw_localization?.triggered && (
            <PipelineVisualizer data={{ ...result.npw_localization, segment_length: params.segment_length }} />
          )}
        </div>
      </div>
    </div>
  );
}
