import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, CXgate, Interferometer, S2gate, MZgate, Rgate
import numpy as np

def default_circuit(prog, wires, weights):
    """
    Constructs a symbolic quantum circuit for the 1P1Qm model.
    
    Args:
        prog: The Strawberry Fields program object.
        wires: Number of wires in the circuit.
        weights: A dictionary containing circuit parameters.  
    """
    # Extract non-trainable parameters from weights
    eta = [weights[f'eta_{i}'] for i in range(wires)]
    pt = [weights[f'pt_{i}'] for i in range(wires)]
    phi = [weights[f'phi_{i}'] for i in range(wires)]

    # Extract trainable parameters
    s_scale = weights['s_scale']    
    squeeze_mag = [weights[f'squeeze_mag_{i}'] for i in range(wires)]
    squeeze_phase = [weights[f'squeeze_phase_{i}'] for i in range(wires)]
    disp_mag = [weights[f'disp_mag_{i}'] for i in range(wires)]
    disp_phase = [weights[f'disp_phase_{i}'] for i in range(wires)]
    
    with prog.context as q:
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        for w in range(wires):
            Sgate(eta[w], pt[w]*phi[w]/2) | q[w] # Encode eta and phi
            Dgate(scale*pt[w], eta[w])    | q[w] # Encode scale, pt, and eta
            
        for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            CXgate(1.0) | (q[a], q[b]) # Entangle wires
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2]) # Further entanglement
            
        for w in range(wires):
            Sgate(squeeze_mag[w], squeeze_phase[w]) | q[w] # Encode trainable parameters
            Dgate(disp_mag[w], disp_phase[w]) | q[w] # Encode trainable parameters
    
    return prog


def new_circuit(prog, wires, weights):
    """Added trainable CX Gate parameters.
    """

    # data constants 
    eta  = [weights[f"eta_{i}"] for i in range(wires)]
    pt   = [weights[f"pt_{i}"]  for i in range(wires)]
    phi  = [weights[f"phi_{i}"] for i in range(wires)]

    # global scale parameter
    s_scale = weights["s_scale"]

    # per‑mode Gaussian trainables
    squeeze_mag   = [weights[f"squeeze_mag_{i}"]   for i in range(wires)]
    squeeze_phase = [weights[f"squeeze_phase_{i}"] for i in range(wires)]
    disp_mag      = [weights[f"disp_mag_{i}"]      for i in range(wires)]
    disp_phase    = [weights[f"disp_phase_{i}"]    for i in range(wires)]

    # CX coupling strengths─
    cx_pairs = [(i, j) for i in range(wires) for j in range(i+1, wires)]
    cx_theta = {(a, b): weights[f"cx_theta_{a}_{b}"] for (a, b) in cx_pairs}

    with prog.context as q:
        # data-encoding layer
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        for w in range(wires):
            Sgate(eta[w], pt[w] * phi[w] / 2) | q[w]
            Dgate(scale * pt[w], eta[w])      | q[w]

        # trainable CX entanglers
        for (a, b) in cx_pairs:
            CXgate(cx_theta[(a, b)]) | (q[a], q[b])
            
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2]) # Further entanglement
            
        for w in range(wires):
            Sgate(squeeze_mag[w], squeeze_phase[w]) | q[w] # Encode trainable parameters
            Dgate(disp_mag[w], disp_phase[w]) | q[w] # Encode trainable parameters
    
    return prog

def x8_circuit(prog, wires, weights):
    """
    X8-compatible version of new_circuit:
    - S2gate instead of Sgate+Dgate pair -- EVEN NUMBER OF MODES REQUIRED!!! 
    - BSgate instead of CXgate (direct substitution)
    - MZgate to approximate Dgate displacement
    - Rgate instead of final Sgate
    
    Args:
        prog: SF Program )
        wires: Number of wires 
        weights: Same weight dictionary as new_circuit
    """
    # Same weights as new_circuit
    # data constants 
    eta  = [weights[f"eta_{i}"] for i in range(wires)]
    pt   = [weights[f"pt_{i}"]  for i in range(wires)]
    phi  = [weights[f"phi_{i}"] for i in range(wires)]

    # global scale parameter
    s_scale = weights["s_scale"]

    # per‑mode Gaussian trainables
    squeeze_mag   = [weights[f"squeeze_mag_{i}"]   for i in range(wires)]
    squeeze_phase = [weights[f"squeeze_phase_{i}"] for i in range(wires)]
    disp_mag      = [weights[f"disp_mag_{i}"]      for i in range(wires)]
    disp_phase    = [weights[f"disp_phase_{i}"]    for i in range(wires)]

    # CX coupling strengths 
    cx_pairs = [(i, j) for i in range(wires) for j in range(i+1, wires)]
    cx_theta = {(a, b): weights[f"cx_theta_{a}_{b}"] for (a, b) in cx_pairs}

    with prog.context as q:
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        
        # Data encoding layer - replace Sgate+Dgate pair with S2gate+MZgate
        for w in range(0, wires, 2):
            if w + 1 < wires:
                # Use S2gate to replace the initial Sgate operations
                r_param = eta[w] 
                phi_param = pt[w] * phi[w] / 2  
                S2gate(r_param, phi_param) | (q[w], q[w+1])
                
        # Approximate Dgate displacement with MZgate 
        for w in range(wires):
            if w < wires - 1:
                # MZgate parameters: internal phase, external phase
                phi_internal = eta[w]  
                phi_external = scale * pt[w]  
                MZgate(phi_internal, phi_external) | (q[w], q[(w+1) % wires])
            
        # Replace CXgate with BSgate (direct substitution - preserves coupling strengths)
        for (a, b) in cx_pairs:
            BSgate(cx_theta[(a, b)], 0) | (q[a], q[b])
            
        # Original BSgate entanglement
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
        # Final layer
        for w in range(wires):
            # Use Rgate with squeeze_mag and squeeze_phase
            # Rgate only takes phase, so we encode magnitude in the phase somehow
            Rgate(squeeze_phase[w] + squeeze_mag[w]) | q[w]  # Combine mag and phase
            
            # Approximate final Dgate with MZgate
            if w < wires - 1:
                phi_internal = disp_phase[w] 
                phi_external = disp_mag[w]  
                MZgate(phi_internal, phi_external) | (q[w], q[(w+1) % wires])
    
    return prog
