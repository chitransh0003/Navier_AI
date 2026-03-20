import { useState, useCallback } from 'react';

export interface AnalysisResult {
  classification: 'SAFE' | 'WARNING' | 'CRITICAL';
  anomaly_score: number;
  scs_report: {
    scs: number;
    flags: string[];
    overall_divergence_pct: number;
    rul_hours: number;
    ambient_temp_c: number;
  };
  physics_report: {
    reynolds_number: number;
    pinn_loss_weight: number;
    mass_residual: number;
    darcy_friction_factor: number;
    is_turbulent: boolean;
  };
  rul_hours: number;
  rul_days: number;
  degradation_rate: string;
  class_probabilities: { SAFE: number; WARNING: number; CRITICAL: number };
  npw_localization: {
    triggered: boolean;
    leak_distance_m: number;
    wave_speed_ms: number;
    delta_t_s: number;
    confidence: number;
    method: string;
  };
  batch_change: { detected: boolean };
  alert_message: string;
  processing_time_ms: number;
  fluid_type: string;
  model_backend: string;
}

const MOCK_RESULT: AnalysisResult = {
  classification: "WARNING",
  anomaly_score: 0.2188,
  scs_report: { scs: 49.9, flags: ["HIGH_PINN_DIVERGENCE", "SENSOR_DRIFT_RISK_HIGH_TEMP", "ACOUSTIC_SPIKE_DETECTED"], overall_divergence_pct: 22.1, rul_hours: 78075, ambient_temp_c: 44.0 },
  physics_report: { reynolds_number: 153461, pinn_loss_weight: 0.677, mass_residual: 0.86, darcy_friction_factor: 0.0168, is_turbulent: true },
  rul_hours: 78075,
  rul_days: 3253,
  degradation_rate: "NORMAL",
  class_probabilities: { SAFE: 0.38, WARNING: 0.49, CRITICAL: 0.13 },
  npw_localization: { triggered: true, leak_distance_m: 3017, wave_speed_ms: 1129, delta_t_s: 3.51, confidence: 0.43, method: "NPW" },
  batch_change: { detected: false },
  alert_message: "WARNING — Anomalous pressure/flow pattern detected. NPW indicates possible anomaly at 3017m.",
  processing_time_ms: 18.4,
  fluid_type: "HSD",
  model_backend: "numpy",
};

export function useNavierAnalysis() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const runAnalysis = useCallback(async (params: Record<string, unknown>) => {
    setIsLoading(true);
    setIsDemoMode(false);
    setResult(null);

    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);

      const res = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!res.ok) throw new Error('API error');
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch {
      await new Promise(r => setTimeout(r, 500));
      setIsDemoMode(true);
      setResult({ ...MOCK_RESULT, fluid_type: (params.fluid_type as string) || 'HSD' });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setIsDemoMode(false);
  }, []);

  return { result, isLoading, isDemoMode, runAnalysis, reset };
}
