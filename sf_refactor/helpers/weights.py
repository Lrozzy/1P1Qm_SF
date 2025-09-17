"""
Centralized weight management for Strawberry Fields circuits.

Responsibilities
- Create symbolic parameters on a given sf.Program for a chosen circuit.
- Create corresponding TF trainable variables for trainable parameters.
- Build runtime args dict mapping symbol names -> tensors for engine.run.
- Save/serialize trained weights to numpy/scalars for persistence.

Supported circuits
- new_circuit: embedding + trainable CX + BS ring + per‑mode Gaussian (squeeze + disp)
- maximally_entangled_circuit: embedding + Clements-like fixed 50:50 mesh + per‑mode Gaussian
- new_entangled_circuit: embedding + Interferometer A (trainable) + Squeeze + Interferometer B (trainable)

Notes
- Symbol naming follows existing conventions in the repo to avoid breaking saved runs.
- For new_entangled_circuit, mesh params use prefixes 'int1' and 'int2':
  int{1,2}_theta_{layer}_{i}_{j}, int{1,2}_phi_{layer}_{i}_{j}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
import strawberryfields as sf


# -------------------------------
# Data classes
# -------------------------------

@dataclass
class SymbolicWeights:
    """Holds SF symbolic params for a circuit and helper metadata."""
    # core
    s_scale: any
    disp_mag: List[any]
    disp_phase: List[any]
    squeeze_mag: List[any]
    squeeze_phase: List[any]
    eta: List[any]
    phi: List[any]
    pt: List[any]

    # optional blocks
    cx_theta: Dict[Tuple[int, int], any] = None
    int1_theta: Dict[Tuple[int, int, int], any] = None  # (layer, i, j)
    int1_phi: Dict[Tuple[int, int, int], any] = None
    int2_theta: Dict[Tuple[int, int, int], any] = None
    int2_phi: Dict[Tuple[int, int, int], any] = None

    def as_weights_dict(self, wires: int) -> Dict[str, any]:
        d: Dict[str, any] = {
            "s_scale": self.s_scale,
            **{f"disp_mag_{w}": self.disp_mag[w] for w in range(wires)},
            **{f"disp_phase_{w}": self.disp_phase[w] for w in range(wires)},
            **{f"squeeze_mag_{w}": self.squeeze_mag[w] for w in range(wires)},
            **{f"squeeze_phase_{w}": self.squeeze_phase[w] for w in range(wires)},
            **{f"eta_{w}": self.eta[w] for w in range(wires)},
            **{f"phi_{w}": self.phi[w] for w in range(wires)},
            **{f"pt_{w}": self.pt[w] for w in range(wires)},
        }
        if self.cx_theta:
            for (a, b), v in self.cx_theta.items():
                d[f"cx_theta_{a}_{b}"] = v
        if self.int1_theta:
            for (layer, i, j), v in self.int1_theta.items():
                d[f"int1_theta_{layer}_{i}_{j}"] = v
        if self.int1_phi:
            for (layer, i, j), v in self.int1_phi.items():
                d[f"int1_phi_{layer}_{i}_{j}"] = v
        if self.int2_theta:
            for (layer, i, j), v in self.int2_theta.items():
                d[f"int2_theta_{layer}_{i}_{j}"] = v
        if self.int2_phi:
            for (layer, i, j), v in self.int2_phi.items():
                d[f"int2_phi_{layer}_{i}_{j}"] = v
        return d


@dataclass
class TrainableVars:
    """Holds TF Variables for trainable parameters; provides export helpers."""
    s_scale: tf.Variable
    disp_mag: List[tf.Variable]
    disp_phase: List[tf.Variable]
    squeeze_mag: List[tf.Variable]
    squeeze_phase: List[tf.Variable]
    cx_theta: Optional[Dict[Tuple[int, int], tf.Variable]] = None
    int1_theta: Optional[Dict[Tuple[int, int, int], tf.Variable]] = None
    int1_phi: Optional[Dict[Tuple[int, int, int], tf.Variable]] = None
    int2_theta: Optional[Dict[Tuple[int, int, int], tf.Variable]] = None
    int2_phi: Optional[Dict[Tuple[int, int, int], tf.Variable]] = None

    def list_vars(self) -> List[tf.Variable]:
        vars_: List[tf.Variable] = [
            self.s_scale,
            *self.disp_mag,
            *self.disp_phase,
            *self.squeeze_mag,
            *self.squeeze_phase,
        ]
        for block in [self.cx_theta, self.int1_theta, self.int1_phi, self.int2_theta, self.int2_phi]:
            if block:
                for k in sorted(block.keys()):
                    vars_.append(block[k])
        return vars_

    def var_names(self, wires: int) -> List[str]:
        names = ["s_scale"]
        if self.disp_mag and len(self.disp_mag) == wires:
            names.extend([f"disp_mag_{i}" for i in range(wires)])
        if self.disp_phase and len(self.disp_phase) == wires:
            names.extend([f"disp_phase_{i}" for i in range(wires)])
        if self.squeeze_mag and len(self.squeeze_mag) == wires:
            names.extend([f"squeeze_mag_{i}" for i in range(wires)])
        if self.squeeze_phase and len(self.squeeze_phase) == wires:
            names.extend([f"squeeze_phase_{i}" for i in range(wires)])
        if self.cx_theta:
            names.extend([f"cx_theta_{a}_{b}" for (a, b) in sorted(self.cx_theta.keys())])
        if self.int1_theta:
            names.extend([f"int1_theta_{L}_{i}_{j}" for (L, i, j) in sorted(self.int1_theta.keys())])
        if self.int1_phi:
            names.extend([f"int1_phi_{L}_{i}_{j}" for (L, i, j) in sorted(self.int1_phi.keys())])
        if self.int2_theta:
            names.extend([f"int2_theta_{L}_{i}_{j}" for (L, i, j) in sorted(self.int2_theta.keys())])
        if self.int2_phi:
            names.extend([f"int2_phi_{L}_{i}_{j}" for (L, i, j) in sorted(self.int2_phi.keys())])
        return names

    def export_numpy(self, wires: int) -> Dict[str, float]:
        out = {"s_scale": self.s_scale.numpy()}
        if self.disp_mag and len(self.disp_mag) == wires:
            for i in range(wires):
                out[f"disp_mag_{i}"] = self.disp_mag[i].numpy()
        if self.disp_phase and len(self.disp_phase) == wires:
            for i in range(wires):
                out[f"disp_phase_{i}"] = self.disp_phase[i].numpy()
        if self.squeeze_mag and len(self.squeeze_mag) == wires:
            for i in range(wires):
                out[f"squeeze_mag_{i}"] = self.squeeze_mag[i].numpy()
        if self.squeeze_phase and len(self.squeeze_phase) == wires:
            for i in range(wires):
                out[f"squeeze_phase_{i}"] = self.squeeze_phase[i].numpy()
        for name, block in [
            ("cx_theta", self.cx_theta),
            ("int1_theta", self.int1_theta),
            ("int1_phi", self.int1_phi),
            ("int2_theta", self.int2_theta),
            ("int2_phi", self.int2_phi),
        ]:
            if block:
                for key, v in block.items():
                    if name == "cx_theta":
                        a, b = key
                        out[f"cx_theta_{a}_{b}"] = v.numpy()
                    else:
                        L, i, j = key
                        out[f"{name}_{L}_{i}_{j}"] = v.numpy()
        return out


# -------------------------------
# Factories
# -------------------------------

def create_symbolic_weights_for_new_circuit(prog: sf.Program, wires: int) -> SymbolicWeights:
    s_scale = prog.params("s_scale")
    disp_mag = [prog.params(f"disp_mag_{w}") for w in range(wires)]
    disp_phase = [prog.params(f"disp_phase_{w}") for w in range(wires)]
    squeeze_mag = [prog.params(f"squeeze_mag_{w}") for w in range(wires)]
    squeeze_phase = [prog.params(f"squeeze_phase_{w}") for w in range(wires)]
    eta = [prog.params(f"eta_{w}") for w in range(wires)]
    phi = [prog.params(f"phi_{w}") for w in range(wires)]
    pt = [prog.params(f"pt_{w}") for w in range(wires)]

    cx_pairs = [(i, j) for i in range(wires) for j in range(i + 1, wires)]
    cx_theta = {(a, b): prog.params(f"cx_theta_{a}_{b}") for (a, b) in cx_pairs}

    return SymbolicWeights(
        s_scale=s_scale,
        disp_mag=disp_mag,
        disp_phase=disp_phase,
        squeeze_mag=squeeze_mag,
        squeeze_phase=squeeze_phase,
        eta=eta,
        phi=phi,
        pt=pt,
        cx_theta=cx_theta,
    )


def create_symbolic_weights_for_max_ent(prog: sf.Program, wires: int) -> SymbolicWeights:
    s_scale = prog.params("s_scale")
    disp_mag = [prog.params(f"disp_mag_{w}") for w in range(wires)]
    disp_phase = [prog.params(f"disp_phase_{w}") for w in range(wires)]
    squeeze_mag = [prog.params(f"squeeze_mag_{w}") for w in range(wires)]
    squeeze_phase = [prog.params(f"squeeze_phase_{w}") for w in range(wires)]
    eta = [prog.params(f"eta_{w}") for w in range(wires)]
    phi = [prog.params(f"phi_{w}") for w in range(wires)]
    pt = [prog.params(f"pt_{w}") for w in range(wires)]

    return SymbolicWeights(
        s_scale=s_scale,
        disp_mag=disp_mag,
        disp_phase=disp_phase,
        squeeze_mag=squeeze_mag,
        squeeze_phase=squeeze_phase,
        eta=eta,
        phi=phi,
        pt=pt,
    )


def create_symbolic_weights_for_new_entangled(prog: sf.Program, wires: int) -> SymbolicWeights:
    """Embedding + two trainable interferometers with separate params."""
    s_scale = prog.params("s_scale")
    # keep disp symbols to preserve saver compatibility though circuit doesn't use them
    disp_mag = [prog.params(f"disp_mag_{w}") for w in range(wires)]
    disp_phase = [prog.params(f"disp_phase_{w}") for w in range(wires)]
    squeeze_mag = [prog.params(f"squeeze_mag_{w}") for w in range(wires)]
    squeeze_phase = [prog.params(f"squeeze_phase_{w}") for w in range(wires)]
    eta = [prog.params(f"eta_{w}") for w in range(wires)]
    phi = [prog.params(f"phi_{w}") for w in range(wires)]
    pt = [prog.params(f"pt_{w}") for w in range(wires)]

    depth = wires
    int1_theta: Dict[Tuple[int, int, int], any] = {}
    int1_phi: Dict[Tuple[int, int, int], any] = {}
    int2_theta: Dict[Tuple[int, int, int], any] = {}
    int2_phi: Dict[Tuple[int, int, int], any] = {}
    for L in range(depth):
        start = 0 if (L % 2 == 0) else 1
        for i in range(start, wires - 1, 2):
            j = i + 1
            int1_theta[(L, i, j)] = prog.params(f"int1_theta_{L}_{i}_{j}")
            int1_phi[(L, i, j)] = prog.params(f"int1_phi_{L}_{i}_{j}")
            int2_theta[(L, i, j)] = prog.params(f"int2_theta_{L}_{i}_{j}")
            int2_phi[(L, i, j)] = prog.params(f"int2_phi_{L}_{i}_{j}")

    return SymbolicWeights(
        s_scale=s_scale,
        disp_mag=disp_mag,
        disp_phase=disp_phase,
        squeeze_mag=squeeze_mag,
        squeeze_phase=squeeze_phase,
        eta=eta,
        phi=phi,
        pt=pt,
        int1_theta=int1_theta,
        int1_phi=int1_phi,
        int2_theta=int2_theta,
        int2_phi=int2_phi,
    )


# -------------------------------
# Trainable TF variable factories
# -------------------------------

def create_trainables_for_new_circuit(wires: int, cx_pairs: Optional[List[Tuple[int, int]]] = None) -> TrainableVars:
    rnd = tf.random_uniform_initializer(-0.1, 0.1)
    s_scale = tf.Variable(rnd(()))
    disp_mag = [tf.Variable(rnd(())) for _ in range(wires)]
    disp_phase = [tf.Variable(rnd(())) for _ in range(wires)]
    squeeze_mag = [tf.Variable(rnd(())) for _ in range(wires)]
    squeeze_phase = [tf.Variable(rnd(())) for _ in range(wires)]
    if cx_pairs is None:
        cx_pairs = [(i, j) for i in range(wires) for j in range(i + 1, wires)]
    cx_theta = {(a, b): tf.Variable(rnd(())) for (a, b) in cx_pairs}
    return TrainableVars(s_scale, disp_mag, disp_phase, squeeze_mag, squeeze_phase, cx_theta=cx_theta)


def create_trainables_for_max_ent(wires: int) -> TrainableVars:
    rnd = tf.random_uniform_initializer(-0.1, 0.1)
    s_scale = tf.Variable(rnd(()))
    disp_mag = [tf.Variable(rnd(())) for _ in range(wires)]
    disp_phase = [tf.Variable(rnd(())) for _ in range(wires)]
    squeeze_mag = [tf.Variable(rnd(())) for _ in range(wires)]
    squeeze_phase = [tf.Variable(rnd(())) for _ in range(wires)]
    return TrainableVars(s_scale, disp_mag, disp_phase, squeeze_mag, squeeze_phase)


def create_trainables_for_new_entangled(wires: int) -> TrainableVars:
    rnd = tf.random_uniform_initializer(-0.1, 0.1)
    s_scale = tf.Variable(rnd(()))
    # omit displacement variables for this circuit (not used)
    disp_mag: List[tf.Variable] = []
    disp_phase: List[tf.Variable] = []
    squeeze_mag = [tf.Variable(rnd(())) for _ in range(wires)]
    squeeze_phase = [tf.Variable(rnd(())) for _ in range(wires)]

    depth = wires
    int1_theta: Dict[Tuple[int, int, int], tf.Variable] = {}
    int1_phi: Dict[Tuple[int, int, int], tf.Variable] = {}
    int2_theta: Dict[Tuple[int, int, int], tf.Variable] = {}
    int2_phi: Dict[Tuple[int, int, int], tf.Variable] = {}
    for L in range(depth):
        start = 0 if (L % 2 == 0) else 1
        for i in range(start, wires - 1, 2):
            j = i + 1
            int1_theta[(L, i, j)] = tf.Variable(rnd(()))
            int1_phi[(L, i, j)] = tf.Variable(rnd(()))
            int2_theta[(L, i, j)] = tf.Variable(rnd(()))
            int2_phi[(L, i, j)] = tf.Variable(rnd(()) )

    return TrainableVars(
        s_scale,
        disp_mag,
        disp_phase,
        squeeze_mag,
        squeeze_phase,
        int1_theta=int1_theta,
        int1_phi=int1_phi,
        int2_theta=int2_theta,
        int2_phi=int2_phi,
    )


# -------------------------------
# Runtime args builder
# -------------------------------

def build_runtime_args(
    wires: int,
    jet_batch: tf.Tensor,
    jet_pt_batch: tf.Tensor,
    sym: SymbolicWeights,
    train: TrainableVars,
) -> Dict[str, tf.Tensor]:
    """Map symbolic names to tensors for a batch.

    Encodes data features (eta, phi, pt) with scaling, and injects trainables.
    """
    # Feature scaling (mirrors previous logic)
    assumed_limits = {
        "pt": [1e-4, 3000.0],
        "eta": [-0.8, 0.8],
        "phi": [-0.8, 0.8],
    }
    feature_limits = {
        "pt": [0.0, 1.0],
        "eta": [-np.pi, np.pi],
        "phi": [-np.pi, np.pi],
    }

    def scale_feature(value, name):
        a_min, a_max = assumed_limits[name]
        f_min, f_max = feature_limits[name]
        return (value - a_min) / (a_max - a_min) * (f_max - f_min) + f_min

    def scale_pt_by_jet(particle_pts, jet_pt):
        # elementwise scaling of per-particle pt by the jet pt for the batch
        # shapes are [batch] / [batch] -> [batch]
        return particle_pts / jet_pt

    squeeze_batch = jet_batch.shape[0] == 1

    d: Dict[str, tf.Tensor] = {"s_scale": train.s_scale}

    if getattr(train, "disp_mag", None) and len(train.disp_mag) == wires:
        for w in range(wires):
            d[f"disp_mag_{w}"] = train.disp_mag[w]
    if getattr(train, "disp_phase", None) and len(train.disp_phase) == wires:
        for w in range(wires):
            d[f"disp_phase_{w}"] = train.disp_phase[w]
    if getattr(train, "squeeze_mag", None) and len(train.squeeze_mag) == wires:
        for w in range(wires):
            d[f"squeeze_mag_{w}"] = train.squeeze_mag[w]
    if getattr(train, "squeeze_phase", None) and len(train.squeeze_phase) == wires:
        for w in range(wires):
            d[f"squeeze_phase_{w}"] = train.squeeze_phase[w]

    for w in range(wires):
        eta_val = scale_feature(jet_batch[:, w, 0], "eta")
        phi_val = scale_feature(jet_batch[:, w, 1], "phi")
        pt_val = scale_pt_by_jet(jet_batch[:, w, 2], jet_pt_batch)
        if squeeze_batch:
            d[f"eta_{w}"] = tf.squeeze(eta_val)
            d[f"phi_{w}"] = tf.squeeze(phi_val)
            d[f"pt_{w}"] = tf.squeeze(pt_val)
        else:
            d[f"eta_{w}"] = eta_val
            d[f"phi_{w}"] = phi_val
            d[f"pt_{w}"] = pt_val

    if train.cx_theta:
        for (a, b), v in train.cx_theta.items():
            d[f"cx_theta_{a}_{b}"] = v

    for name, block in [
        ("int1_theta", train.int1_theta),
        ("int1_phi", train.int1_phi),
        ("int2_theta", train.int2_theta),
        ("int2_phi", train.int2_phi),
    ]:
        if block:
            for (L, i, j), v in block.items():
                d[f"{name}_{L}_{i}_{j}"] = v

    return d


# -------------------------------
# High-level wiring for scripts
# -------------------------------

class CircuitKind:
    NEW = "new_circuit"
    MAX_ENT = "maximally_entangled_circuit"
    NEW_ENT = "new_entangled_circuit"


def make_symbolic_and_circuit(
    prog: sf.Program,
    wires: int,
    which: str,
    circuits_module,
):
    """Return (symbolic_weights, circuit_fn) for selected circuit."""
    if which == CircuitKind.NEW:
        sym = create_symbolic_weights_for_new_circuit(prog, wires)
        circuit_fn = circuits_module.new_circuit
    elif which == CircuitKind.MAX_ENT:
        sym = create_symbolic_weights_for_max_ent(prog, wires)
        circuit_fn = circuits_module.maximally_entangled_circuit
    elif which == CircuitKind.NEW_ENT:
        sym = create_symbolic_weights_for_new_entangled(prog, wires)
        circuit_fn = circuits_module.new_entangled_circuit
    else:
        raise ValueError(f"Unknown circuit kind: {which}")
    return sym, circuit_fn


def make_trainables(wires: int, which: str) -> TrainableVars:
    if which == CircuitKind.NEW:
        return create_trainables_for_new_circuit(wires)
    elif which == CircuitKind.MAX_ENT:
        return create_trainables_for_max_ent(wires)
    elif which == CircuitKind.NEW_ENT:
        return create_trainables_for_new_entangled(wires)
    else:
        raise ValueError(f"Unknown circuit kind: {which}")