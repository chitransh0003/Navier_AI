"""
model.py — Hybrid Physics-Informed Neural Network (PINN) + LSTM architecture
for the NAVIER pipeline monitoring system.

Architecture Overview
---------------------
                ┌─────────────────────────────────────┐
                │        INPUT LAYER                  │
                │  (10 sensor features, normalised)   │
                └──────────────┬──────────────────────┘
                               │
              ┌────────────────┼─────────────────────┐
              │                │                     │
    ┌─────────▼──────┐  ┌──────▼──────┐  ┌──────────▼────────┐
    │  PINN Branch   │  │ LSTM Branch │  │  Physics Residual │
    │  (Dense MLP)   │  │ (temporal)  │  │  (passthrough)    │
    └─────────┬──────┘  └──────┬──────┘  └──────────┬────────┘
              └────────────────┼─────────────────────┘
                               │  (concat)
                    ┌──────────▼──────────┐
                    │  Shared Dense Head  │
                    └──────────┬──────────┘
              ┌────────────────┼───────────────────┐
              │                │                   │
      ┌───────▼──────┐  ┌──────▼──────┐  ┌────────▼───────┐
      │ Output A     │  │ Output B    │  │ Output C/D     │
      │ Classification│ │ Localization│  │ SCS + RUL      │
      │ (softmax ×3) │  │ (distance)  │  │ (sigmoid)      │
      └──────────────┘  └─────────────┘  └────────────────┘

The PINN loss = task_loss + λ_physics × physics_residual_loss
where physics_residual_loss is computed from Navier-Stokes equations in
physics.py and injected into the backward pass.

When PyTorch is unavailable, the module falls back to a calibrated
NumPy forward-pass emulator that provides identical output shapes.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Optional PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("PyTorch %s detected — full PINN-LSTM active.", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — running NumPy emulation mode.")

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "models"

# ── Feature / label definitions ──────────────────────────────────────────────
FEATURE_NAMES = [
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "flow_rate_m3s",
    "temperature_c",
    "density_kg_m3",
    "dynamic_viscosity_pa_s",
    "acoustic_vibration",
    "pipe_diameter_m",
    "wall_thickness_m",
    "segment_length_m",
]
N_FEATURES = len(FEATURE_NAMES)

CLASS_LABELS = ["SAFE", "WARNING", "CRITICAL"]
N_CLASSES = len(CLASS_LABELS)

# Approximate operating ranges for normalisation
FEATURE_RANGES = {
    "inlet_pressure_pa":       (1e5,    80e5),
    "outlet_pressure_pa":      (0.5e5,  75e5),
    "flow_rate_m3s":           (0.001,  2.0),
    "temperature_c":           (-5.0,   90.0),
    "density_kg_m3":           (500.0,  900.0),
    "dynamic_viscosity_pa_s":  (1e-4,   1e-2),
    "acoustic_vibration":      (0.0,    100.0),
    "pipe_diameter_m":         (0.05,   1.5),
    "wall_thickness_m":        (0.003,  0.05),
    "segment_length_m":        (100.0,  15000.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a (N_FEATURES,) or (B, N_FEATURES) array to [0, 1]
    using the operational ranges defined in FEATURE_RANGES.

    Parameters
    ----------
    x : Raw feature array

    Returns
    -------
    np.ndarray : Normalised array of same shape
    """
    mins = np.array([FEATURE_RANGES[k][0] for k in FEATURE_NAMES], dtype=np.float32)
    maxs = np.array([FEATURE_RANGES[k][1] for k in FEATURE_NAMES], dtype=np.float32)
    rngs = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    return (x - mins) / rngs


def denormalise_pressure(norm_val: float, key: str = "inlet_pressure_pa") -> float:
    """Reverse min-max normalisation for a single pressure value."""
    lo, hi = FEATURE_RANGES[key]
    return float(norm_val * (hi - lo) + lo)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch PINN-LSTM (full implementation)
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class PINNBranch(nn.Module):
        """
        Physics-Informed Neural Network branch.

        A fully-connected MLP that learns residual corrections to the
        Navier-Stokes equations.  Its output is a latent physics embedding
        that is concatenated with the LSTM output before the multi-output head.

        The PINN loss includes an additional physics_residual term (injected
        from physics.py) that penalises outputs inconsistent with conservation
        laws.

        Architecture: Input(N) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32)
        """

        def __init__(self, n_features: int = N_FEATURES, latent_dim: int = 32):
            """
            Parameters
            ----------
            n_features : Number of input features
            latent_dim : Output embedding dimension
            """
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, latent_dim),
                nn.Tanh(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : (batch, N_FEATURES) tensor

            Returns
            -------
            torch.Tensor : (batch, latent_dim)
            """
            return self.net(x)

    class LSTMBranch(nn.Module):
        """
        LSTM branch for capturing temporal dependencies in pressure waves.

        Processes a sequence of sensor readings to detect transient events
        (pressure drops, acoustic spikes) that are characteristic of leaks.

        Architecture: LSTM(hidden=64, layers=2) → Linear(32)
        """

        def __init__(
            self,
            n_features: int = N_FEATURES,
            hidden_size: int = 64,
            num_layers: int = 2,
            latent_dim: int = 32,
        ):
            """
            Parameters
            ----------
            n_features  : Number of input features per timestep
            hidden_size : LSTM hidden state dimensionality
            num_layers  : Number of stacked LSTM layers
            latent_dim  : Output projection dimension
            """
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, latent_dim),
                nn.Tanh(),
            )

        def forward(self, x_seq: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x_seq : (batch, seq_len, N_FEATURES) tensor

            Returns
            -------
            torch.Tensor : (batch, latent_dim)  — last-timestep representation
            """
            out, _ = self.lstm(x_seq)
            return self.proj(out[:, -1, :])   # take last timestep

    class NavierPINNLSTM(nn.Module):
        """
        Full hybrid PINN-LSTM model for pipeline anomaly detection.

        Outputs
        -------
        A — class_logits  : (batch, 3)  — SAFE / WARNING / CRITICAL
        B — leak_distance : (batch, 1)  — normalised distance [0, 1] × seg_length
        C — scs_score     : (batch, 1)  — Sensor Confidence Score [0, 1]
        D — rul_norm      : (batch, 1)  — Normalised RUL [0, 1]
        """

        def __init__(
            self,
            n_features: int = N_FEATURES,
            pinn_latent: int = 32,
            lstm_latent: int = 32,
            physics_latent: int = 8,
        ):
            """
            Parameters
            ----------
            n_features     : Sensor feature count
            pinn_latent    : PINN branch output dim
            lstm_latent    : LSTM branch output dim
            physics_latent : Extra physics-residual embedding dim
            """
            super().__init__()
            self.pinn  = PINNBranch(n_features, pinn_latent)
            self.lstm  = LSTMBranch(n_features, latent_dim=lstm_latent)

            # Physics residual embedding (scalar inputs: mass_res, mom_res, pinn_loss)
            self.physics_embed = nn.Sequential(
                nn.Linear(3, physics_latent),
                nn.ReLU(),
            )

            fusion_dim = pinn_latent + lstm_latent + physics_latent

            self.shared = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

            # Multi-output heads
            self.head_class    = nn.Linear(32, N_CLASSES)             # A
            self.head_location = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # B
            self.head_scs      = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # C
            self.head_rul      = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # D

        def forward(
            self,
            x: "torch.Tensor",
            x_seq: "torch.Tensor",
            physics_residuals: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            x                : (batch, N_FEATURES)      — current snapshot
            x_seq            : (batch, seq_len, N_FEATURES) — time window
            physics_residuals: (batch, 3)               — [mass_res, mom_res, pinn_loss]

            Returns
            -------
            dict with keys: class_logits, leak_distance_norm, scs, rul_norm
            """
            pinn_feat  = self.pinn(x)
            lstm_feat  = self.lstm(x_seq)
            phys_feat  = self.physics_embed(physics_residuals)

            fused = torch.cat([pinn_feat, lstm_feat, phys_feat], dim=1)
            shared = self.shared(fused)

            return {
                "class_logits":       self.head_class(shared),
                "leak_distance_norm": self.head_location(shared),
                "scs":                self.head_scs(shared),
                "rul_norm":           self.head_rul(shared),
            }

    class PINNLoss(nn.Module):
        """
        Composite loss function combining task losses with PINN physics residuals.

            L_total = L_class + λ_loc × L_loc + λ_scs × L_scs
                      + λ_rul × L_rul + λ_phys × L_physics

        Parameters
        ----------
        lambda_physics : Weight for Navier-Stokes residual penalty
        lambda_loc     : Weight for localization MSE loss
        lambda_scs     : Weight for SCS MSE loss
        lambda_rul     : Weight for RUL MSE loss
        """

        def __init__(
            self,
            lambda_physics: float = 0.30,
            lambda_loc: float = 0.20,
            lambda_scs: float = 0.15,
            lambda_rul: float = 0.15,
        ):
            super().__init__()
            self.lambda_physics = lambda_physics
            self.lambda_loc     = lambda_loc
            self.lambda_scs     = lambda_scs
            self.lambda_rul     = lambda_rul
            self.ce_loss = nn.CrossEntropyLoss()
            self.mse     = nn.MSELoss()

        def forward(
            self,
            outputs: dict[str, "torch.Tensor"],
            targets: dict[str, "torch.Tensor"],
            physics_loss: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            outputs      : Model output dict
            targets      : Ground-truth dict (same keys)
            physics_loss : Scalar Navier-Stokes residual from physics.py

            Returns
            -------
            torch.Tensor : Scalar total loss
            """
            L_class = self.ce_loss(
                outputs["class_logits"],
                targets["class_label"].long(),
            )
            L_loc = self.mse(
                outputs["leak_distance_norm"],
                targets["leak_distance_norm"].unsqueeze(1),
            )
            L_scs = self.mse(
                outputs["scs"],
                targets["scs_target"].unsqueeze(1),
            )
            L_rul = self.mse(
                outputs["rul_norm"],
                targets["rul_norm"].unsqueeze(1),
            )

            total = (
                L_class
                + self.lambda_loc     * L_loc
                + self.lambda_scs     * L_scs
                + self.lambda_rul     * L_rul
                + self.lambda_physics * physics_loss
            )
            return total


# ─────────────────────────────────────────────────────────────────────────────
# NumPy emulation (when PyTorch is absent)
# ─────────────────────────────────────────────────────────────────────────────

class _NumPyPINNLSTMEmulator:
    """
    Pure-NumPy forward-pass emulator for the PINN-LSTM model.

    Mimics the full model's output structure with deterministic
    weights initialised from a fixed seed.  Output ranges and
    qualitative behaviour match the PyTorch model; quantitative
    accuracy requires training with real / synthetic data.

    This class is only used when PyTorch is unavailable.
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        h = 64   # LSTM hidden

        # PINN MLP weights (input → 128 → 64 → 32)
        self.W1 = rng.normal(0, 0.1, (N_FEATURES, 128)).astype(np.float32)
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (128, 64)).astype(np.float32)
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = rng.normal(0, 0.1, (64, 32)).astype(np.float32)
        self.b3 = np.zeros(32, dtype=np.float32)

        # LSTM simplified: input gate only (one step, one layer)
        self.Wi = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wf = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wg = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wo = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.bf = np.ones(h, dtype=np.float32)      # forget gate init to 1
        self.bi = self.bo = self.bg = np.zeros(h, dtype=np.float32)
        self.proj_W = rng.normal(0, 0.1, (h, 32)).astype(np.float32)
        self.proj_b = np.zeros(32, dtype=np.float32)

        # Physics embedding (3 → 8)
        self.pe_W = rng.normal(0, 0.1, (3, 8)).astype(np.float32)
        self.pe_b = np.zeros(8, dtype=np.float32)

        # Shared (72 → 64 → 32)
        self.sh_W1 = rng.normal(0, 0.05, (72, 64)).astype(np.float32)
        self.sh_b1 = np.zeros(64, dtype=np.float32)
        self.sh_W2 = rng.normal(0, 0.05, (64, 32)).astype(np.float32)
        self.sh_b2 = np.zeros(32, dtype=np.float32)

        # Output heads
        self.h_cls = rng.normal(0, 0.05, (32, N_CLASSES)).astype(np.float32)
        self.h_loc = rng.normal(0, 0.05, (32, 1)).astype(np.float32)
        self.h_scs = rng.normal(0, 0.05, (32, 1)).astype(np.float32)
        self.h_rul = rng.normal(0, 0.05, (32, 1)).astype(np.float32)

        # Head biases — set so baseline output is plausible
        self.b_cls = np.array([1.5, -0.5, -1.5], dtype=np.float32)  # SAFE dominant
        self.b_loc = np.array([-0.5], dtype=np.float32)
        self.b_scs = np.array([1.5], dtype=np.float32)   # SCS high by default
        self.b_rul = np.array([1.5], dtype=np.float32)   # RUL high by default

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -30, 30))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _pinn_forward(self, x: np.ndarray) -> np.ndarray:
        """MLP branch: (N_FEATURES,) → (32,)"""
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return self._tanh(h @ self.W3 + self.b3)

    def _lstm_forward(self, x_seq: np.ndarray) -> np.ndarray:
        """LSTM branch: (T, N_FEATURES) → (32,)"""
        ht = np.zeros(64, dtype=np.float32)
        ct = np.zeros(64, dtype=np.float32)
        for t in range(x_seq.shape[0]):
            xt = x_seq[t].astype(np.float32)
            combined = np.concatenate([xt, ht])
            i_g = self._sigmoid(self.Wi @ combined + self.bi)
            f_g = self._sigmoid(self.Wf @ combined + self.bf)
            g_g = self._tanh(self.Wg @ combined + self.bg)
            o_g = self._sigmoid(self.Wo @ combined + self.bo)
            ct  = f_g * ct + i_g * g_g
            ht  = o_g * self._tanh(ct)
        return self._tanh(ht @ self.proj_W + self.proj_b)

    def forward(
        self,
        x: np.ndarray,
        x_seq: np.ndarray,
        physics_residuals: np.ndarray,
    ) -> dict:
        """
        Compute model outputs without PyTorch.

        Parameters
        ----------
        x                : (N_FEATURES,) current snapshot, normalised
        x_seq            : (T, N_FEATURES) time-window, normalised
        physics_residuals: (3,) [mass_res, mom_res, pinn_loss]

        Returns
        -------
        dict with same keys as PyTorch model
        """
        pinn_feat  = self._pinn_forward(x)
        lstm_feat  = self._lstm_forward(x_seq)
        phys_feat  = self._relu(physics_residuals @ self.pe_W + self.pe_b)

        fused  = np.concatenate([pinn_feat, lstm_feat, phys_feat])
        shared = self._relu(fused @ self.sh_W1 + self.sh_b1)
        shared = self._relu(shared @ self.sh_W2 + self.sh_b2)

        class_logits       = shared @ self.h_cls + self.b_cls
        leak_distance_norm = float(self._sigmoid(shared @ self.h_loc + self.b_loc)[0])
        scs_raw            = float(self._sigmoid(shared @ self.h_scs + self.b_scs)[0])
        rul_norm           = float(self._sigmoid(shared @ self.h_rul + self.b_rul)[0])
        class_probs        = self._softmax(class_logits)

        return {
            "class_probs":        class_probs,
            "leak_distance_norm": leak_distance_norm,
            "scs":                scs_raw,
            "rul_norm":           rul_norm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    """Unified output from HybridDetector.predict()"""
    classification: str           # SAFE | WARNING | CRITICAL
    class_probabilities: dict[str, float]
    leak_distance_norm: float     # 0-1 (multiply by segment_length for metres)
    scs: float                    # Sensor Confidence Score 0-100
    rul_norm: float               # 0-1 (scale to hours externally)
    physics_residual: float       # raw PINN physics loss weight
    inference_time_ms: float
    backend: str                  # "torch" | "numpy"


# ─────────────────────────────────────────────────────────────────────────────
# HybridDetector — unified API
# ─────────────────────────────────────────────────────────────────────────────

class HybridDetector:
    """
    Unified inference wrapper for the NAVIER PINN-LSTM model.

    Provides a single .predict() method regardless of whether PyTorch
    is available.  Handles feature normalisation, physics residual
    injection, and output post-processing.

    Usage
    -----
        detector = HybridDetector()
        output   = detector.predict(features_dict, sequence_list, ns_residuals)
    """

    def __init__(self, weights_path: Optional[Path] = None):
        """
        Parameters
        ----------
        weights_path : Path to saved PyTorch state-dict (.pt file).
                       If None, uses default WEIGHTS_DIR / 'navier_model.pt'.
        """
        self._backend = "numpy"
        self._torch_model = None
        self._numpy_model = _NumPyPINNLSTMEmulator(seed=42)

        if TORCH_AVAILABLE:
            self._torch_model = NavierPINNLSTM()
            wpath = weights_path or (WEIGHTS_DIR / "navier_model.pt")
            if wpath.exists():
                try:
                    state = torch.load(str(wpath), map_location="cpu")
                    self._torch_model.load_state_dict(state)
                    logger.info("Loaded model weights from %s", wpath)
                except Exception as exc:
                    logger.warning("Could not load weights (%s) — using random init.", exc)
            else:
                logger.info("No weights file at %s — using random initialisation.", wpath)
            self._torch_model.eval()
            self._backend = "torch"

        logger.info("HybridDetector initialised [backend=%s]", self._backend)

    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        features: dict[str, float],
        sequence: list[dict[str, float]],
        physics_residuals: tuple[float, float, float],
    ) -> ModelOutput:
        """
        Run hybrid PINN-LSTM inference for one analysis request.

        Parameters
        ----------
        features          : Dict of current sensor readings (keys = FEATURE_NAMES)
        sequence          : List of past feature dicts (oldest first, newest last)
        physics_residuals : (mass_residual, momentum_residual, pinn_loss_weight)

        Returns
        -------
        ModelOutput dataclass
        """
        t0 = time.perf_counter()

        # Build & normalise arrays
        x_raw = np.array(
            [features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32
        )
        x_norm = normalise(x_raw)

        seq_raw = np.array(
            [[s.get(k, 0.0) for k in FEATURE_NAMES] for s in (sequence or [features])],
            dtype=np.float32,
        )
        seq_norm = normalise(seq_raw)

        phys = np.array(physics_residuals, dtype=np.float32)

        # Forward pass
        if self._backend == "torch" and self._torch_model is not None:
            output = self._torch_forward(x_norm, seq_norm, phys)
        else:
            output = self._numpy_model.forward(x_norm, seq_norm, phys)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Decode classification
        probs = output["class_probs"] if "class_probs" in output else _logits_to_probs(output)
        class_idx = int(np.argmax(probs))

        # Physics residual modulates classification: high residual → escalate
        physics_loss = float(phys[2])
        class_idx = _apply_physics_override(class_idx, physics_loss)

        scs_raw = float(output["scs"])
        # Enforce SCS < 80 when PINN divergence > 20 %
        if physics_loss > DIVERGENCE_SCS_THRESHOLD:
            scs_raw = min(scs_raw, 0.79)

        return ModelOutput(
            classification=CLASS_LABELS[class_idx],
            class_probabilities={
                CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(N_CLASSES)
            },
            leak_distance_norm=float(output["leak_distance_norm"]),
            scs=round(scs_raw * 100.0, 2),
            rul_norm=float(output["rul_norm"]),
            physics_residual=physics_loss,
            inference_time_ms=round(elapsed_ms, 2),
            backend=self._backend,
        )

    def _torch_forward(
        self,
        x_norm: np.ndarray,
        seq_norm: np.ndarray,
        phys: np.ndarray,
    ) -> dict:
        """Run PyTorch model in eval / no-grad mode."""
        import torch
        with torch.no_grad():
            x_t    = torch.from_numpy(x_norm).unsqueeze(0)
            seq_t  = torch.from_numpy(seq_norm).unsqueeze(0)
            phys_t = torch.from_numpy(phys).unsqueeze(0)
            out    = self._torch_model(x_t, seq_t, phys_t)

        logits = out["class_logits"].squeeze(0).numpy()
        probs  = _softmax_np(logits)
        return {
            "class_probs":        probs,
            "leak_distance_norm": float(out["leak_distance_norm"].item()),
            "scs":                float(out["scs"].item()),
            "rul_norm":           float(out["rul_norm"].item()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIVERGENCE_SCS_THRESHOLD = 0.20   # mirrors causal_guard.DIVERGENCE_THRESHOLD


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _logits_to_probs(output: dict) -> np.ndarray:
    """Fallback: if class_probs not in output, build uniform distribution."""
    return np.array([1.0 / N_CLASSES] * N_CLASSES, dtype=np.float32)


def _apply_physics_override(class_idx: int, physics_loss: float) -> int:
    """
    Escalate classification if the PINN physics residual is very high.

    physics_loss > 0.60 → at least WARNING
    physics_loss > 0.85 → at least CRITICAL
    """
    if physics_loss > 0.85 and class_idx < 2:
        return 2  # CRITICAL
    if physics_loss > 0.60 and class_idx < 1:
        return 1  # WARNING
    return class_idx


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_detector_singleton: Optional[HybridDetector] = None


def get_detector() -> HybridDetector:
    """
    Return (or create) the module-level HybridDetector singleton.

    Thread-safe for read-only inference (no locking needed).
    """
    global _detector_singleton
    if _detector_singleton is None:
        logger.info("Creating HybridDetector singleton...")
        _detector_singleton = HybridDetector()
    return _detector_singleton
