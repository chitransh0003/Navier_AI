import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Zap, Waves, Radio, Shield, Github, Twitter, Linkedin, ArrowRight } from 'lucide-react';
import NavierLogo from '../components/NavierLogo';
import { useState, useEffect } from 'react';

const stagger = { hidden: {}, visible: { transition: { staggerChildren: 0.15 } } };
const fadeUp = { hidden: { opacity: 0, y: 40 }, visible: { opacity: 1, y: 0, transition: { duration: 0.6 } } };

function PipelineBackground() {
  const pipes = [
    { y: '20%', dur: 8 },
    { y: '40%', dur: 12 },
    { y: '60%', dur: 15 },
    { y: '80%', dur: 10 },
  ];
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <svg className="w-full h-full" preserveAspectRatio="none">
        {pipes.map((p, i) => (
          <g key={i}>
            <line x1="0" y1={p.y} x2="100%" y2={p.y} stroke="#CBD5E1" strokeWidth="1.5" />
            <circle r="4" fill="#93C5FD" className={`animate-flow-${i + 1}`} style={{ filter: 'drop-shadow(0 0 8px #3B82F6)' }}>
              <animateMotion dur={`${p.dur}s`} repeatCount="indefinite" path={`M-20,0 L2000,0`} />
              <set attributeName="cy" to={p.y} />
            </circle>
          </g>
        ))}
      </svg>
      {pipes.map((p, i) => (
        <div
          key={`dot-${i}`}
          className="absolute w-2 h-2 rounded-full animate-pulse-glow"
          style={{
            top: p.y,
            background: '#93C5FD',
            boxShadow: '0 0 8px #3B82F6',
            animation: `flow-dot ${p.dur}s linear infinite ${i * 1.5}s`,
          }}
        />
      ))}
    </div>
  );
}

function DemoStrip() {
  const [leakDetected, setLeakDetected] = useState(false);
  const [dotPos, setDotPos] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setLeakDetected(false);
      setDotPos(0);
      let frame = 0;
      const animate = () => {
        frame++;
        const progress = Math.min(frame / 100, 1);
        setDotPos(progress);
        if (progress >= 0.6 && !leakDetected) {
          setLeakDetected(true);
        }
        if (frame < 100) requestAnimationFrame(animate);
      };
      animate();
    }, 6000);
    // Run once immediately
    let frame = 0;
    const animate = () => {
      frame++;
      const progress = Math.min(frame / 100, 1);
      setDotPos(progress);
      if (progress >= 0.6) setLeakDetected(true);
      if (frame < 100) requestAnimationFrame(animate);
    };
    animate();
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="relative w-full h-28">
      <svg className="w-full h-full" viewBox="0 0 1200 120" preserveAspectRatio="none">
        <rect x="0" y="52" width="1200" height="16" rx="8" fill="#E2E8F0" />
        <rect x="0" y="54" width="1200" height="12" rx="6" fill="#F1F5F9" />
        {/* Flow dots */}
        {[0.2, 0.5, 0.8].map((offset, i) => {
          const x = ((dotPos + offset) % 1) * 1200;
          const isLeakDot = i === 0 && dotPos >= 0.55;
          return (
            <g key={i}>
              <circle cx={x} cy="60" r={isLeakDot ? 6 : 5} fill={isLeakDot ? '#EF4444' : '#0A4FD4'} opacity={isLeakDot ? 1 : 0.8}>
                {isLeakDot && <animate attributeName="r" values="6;9;6" dur="0.8s" repeatCount="indefinite" />}
              </circle>
              {isLeakDot && leakDetected && (
                <>
                  <circle cx={x} cy="60" r="12" fill="none" stroke="#EF4444" strokeWidth="2" opacity="0.4" className="animate-ping-leak" />
                  <rect x={x - 45} y="18" width="90" height="24" rx="12" fill="#FEF2F2" stroke="#FECACA" strokeWidth="1" />
                  <text x={x} y="34" textAnchor="middle" fill="#DC2626" fontSize="10" fontWeight="600">Leak detected</text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <section className="relative min-h-screen flex flex-col">
        <PipelineBackground />
        {/* Top bar */}
        <div className="relative z-10 flex items-center justify-between px-8 py-4">
          <NavierLogo />
          <button className="px-4 py-1.5 rounded-full border border-primary text-primary text-sm font-medium hover:bg-primary/5 transition-colors">
            Sign In
          </button>
        </div>
        {/* Center content */}
        <div className="relative z-10 flex-1 flex items-center justify-center">
          <motion.div variants={stagger} initial="hidden" animate="visible" className="text-center max-w-3xl mx-auto px-6">
            <motion.div variants={fadeUp} className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/5 border border-primary/20 text-primary text-sm font-medium mb-8">
              <Zap size={14} />
              Powered by PINN-LSTM Neural Architecture
            </motion.div>
            <motion.h1 variants={fadeUp} className="font-heading text-6xl md:text-7xl font-bold gradient-text leading-tight mb-6">
              The Future of Pipeline Intelligence
            </motion.h1>
            <motion.p variants={fadeUp} className="text-lg text-muted-foreground max-w-xl mx-auto mb-8">
              NAVIER uses Physics-Informed Neural Networks and LSTM time-series analysis to detect leaks, localize breach points, and estimate pipeline health in real time.
            </motion.p>
            <motion.div variants={fadeUp} className="flex items-center justify-center gap-6 mb-8">
              {[
                { icon: Shield, text: '99.7% Accuracy' },
                { icon: Zap, text: '< 3ms Latency' },
                { icon: Waves, text: '5 Fluid Types' },
              ].map((badge, i) => (
                <div key={i} className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <badge.icon size={14} className="text-primary" />
                  {badge.text}
                </div>
              ))}
            </motion.div>
            <motion.div variants={fadeUp}>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate('/dashboard')}
                className="gradient-bg text-primary-foreground font-bold text-lg rounded-full px-10 py-4 shadow-lg hover:shadow-[0_0_30px_rgba(0,201,200,0.4)] transition-shadow"
              >
                Launch Dashboard <ArrowRight className="inline ml-2" size={20} />
              </motion.button>
              <p className="text-xs text-muted-foreground mt-4">No setup required · Demo mode available</p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Feature Highlights */}
      <section className="py-24 px-8 bg-background">
        <h2 className="font-heading text-4xl font-bold text-center text-foreground mb-16">Built on Real Physics</h2>
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {[
            { icon: Waves, color: 'bg-primary/10 text-primary', title: 'Navier-Stokes Engine', desc: '1-D mass and momentum conservation equations compute physics residuals at every timestep, forming the backbone of PINN loss.' },
            { icon: Radio, color: 'bg-secondary/10 text-secondary', title: 'Negative Pressure Wave', desc: 'When a leak occurs, NPW propagates at fluid wave speed. NAVIER localises the breach point using X = (L − a·Δt) / 2.' },
            { icon: Shield, color: 'bg-navier-safe/10 text-navier-safe', title: 'Sensor Trust Score', desc: 'Sensor Confidence Score (SCS) compares PINN-predicted physical truth against real sensor readings. If divergence exceeds 20%, SCS drops below 80%.' },
          ].map((card, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.15 }}
              viewport={{ once: true }}
              className="navier-card text-center"
            >
              <div className={`w-14 h-14 rounded-2xl ${card.color} flex items-center justify-center mx-auto mb-5`}>
                <card.icon size={24} />
              </div>
              <h3 className="font-heading text-lg font-bold text-foreground mb-3">{card.title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{card.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Demo Strip */}
      <section className="py-16 px-8 bg-primary/5">
        <h2 className="font-heading text-3xl font-bold text-center text-foreground mb-10">Watch NAVIER Think</h2>
        <div className="max-w-5xl mx-auto">
          <DemoStrip />
        </div>
      </section>

      {/* Metrics Row */}
      <section className="py-20 px-8 bg-background">
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {[
            { value: '10', label: 'parameters monitored per reading' },
            { value: '3', label: 'detection modes: NPW, Mass Balance, AI' },
            { value: '< 500ms', label: 'from sensor to alert' },
            { value: '5', label: 'petroleum fluids supported' },
          ].map((m, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }} viewport={{ once: true }}>
              <div className="font-heading text-4xl font-bold gradient-text mb-2">{m.value}</div>
              <p className="text-sm text-muted-foreground">{m.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20 px-8 bg-navier-light-bg">
        <div className="text-center max-w-xl mx-auto">
          <h2 className="font-heading text-3xl font-bold text-foreground mb-4">Ready to Monitor Your Pipeline?</h2>
          <p className="text-muted-foreground mb-8">Connect NAVIER to your SCADA system and get intelligent alerts in minutes.</p>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => navigate('/dashboard')}
            className="gradient-bg text-primary-foreground font-bold rounded-full px-10 py-4 shadow-lg hover:shadow-[0_0_30px_rgba(0,201,200,0.4)] transition-shadow"
          >
            Get Started Free <ArrowRight className="inline ml-1" size={18} />
          </motion.button>
          <div className="flex items-center justify-center gap-4 mt-6 text-xs text-muted-foreground">
            {['React', 'FastAPI', 'PINN', 'LSTM'].map(t => (
              <span key={t} className="px-2 py-1 rounded bg-background border border-border">{t}</span>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-6 px-8 flex items-center justify-between bg-background">
        <NavierLogo size="sm" />
        <p className="text-sm text-muted-foreground">© 2025 NAVIER AI. All rights reserved.</p>
        <div className="flex items-center gap-4 text-muted-foreground">
          <a href="#"><Github size={18} /></a>
          <a href="#"><Twitter size={18} /></a>
          <a href="#"><Linkedin size={18} /></a>
        </div>
      </footer>
    </div>
  );
}
