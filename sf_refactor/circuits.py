import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, CXgate, Interferometer, S2gate, MZgate, Rgate, MeasureFock
from strawberryfields.utils import operation
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
            Dgate(scale*pt[w], eta[w])    | q[w] # Encode scale, pt, and eta
            Sgate(eta[w], pt[w]*phi[w]/2) | q[w] # Encode eta and phi
            
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
            Dgate(scale * pt[w], eta[w])      | q[w]
            Sgate(eta[w], pt[w] * phi[w] / 2) | q[w]

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


def multiuploading_circuit(prog, wires, weights, particles_per_wire=2, particle_mapping="interleaved"):
    """
    Circuit with multiuploading encoding - multiple particles encoded sequentially on each qumode.
    Multiuploading!
    
    Args:
        prog: The Strawberry Fields program object.
        wires: Number of wires in the circuit.
        weights: A dictionary containing circuit parameters for all particles.
        particles_per_wire: Number of particles to encode on each wire.
        particle_mapping: "sequential" or "interleaved" - how to map particles to wires
                         sequential: wire0:[0,1], wire1:[2,3], wire2:[4,5] etc.
                         interleaved: wire0:[0,wires], wire1:[1,wires+1], wire2:[2,wires+2] etc.
    """
    
    # Extract trainable parameters (same as new_circuit)
    s_scale = weights['s_scale']    
    squeeze_mag = [weights[f'squeeze_mag_{i}'] for i in range(wires)]
    squeeze_phase = [weights[f'squeeze_phase_{i}'] for i in range(wires)]
    disp_mag = [weights[f'disp_mag_{i}'] for i in range(wires)]
    disp_phase = [weights[f'disp_phase_{i}'] for i in range(wires)]
    
    # CX coupling strengths 
    cx_pairs = [(i, j) for i in range(wires) for j in range(i+1, wires)]
    cx_theta = {(a, b): weights[f"cx_theta_{a}_{b}"] for (a, b) in cx_pairs}

    with prog.context as q:
        # Data-encoding layer: sequentially encode multiple particles per wire
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        
        for w in range(wires):
            # Loop through particles assigned to this wire
            for p in range(particles_per_wire):
                # Calculate particle index based on mapping pattern
                if particle_mapping == "sequential":
                    particle_idx = w * particles_per_wire + p
                elif particle_mapping == "interleaved":
                    particle_idx = p * wires + w
                else:
                    raise ValueError(f"Unknown particle_mapping: {particle_mapping}. Must be 'sequential' or 'interleaved'")
                
                # Extract features for the specific particle
                eta_p = weights[f"eta_{particle_idx}"]
                pt_p = weights[f"pt_{particle_idx}"]
                phi_p = weights[f"phi_{particle_idx}"]

                # Apply encoding gates for this particle
                Dgate(scale * pt_p, eta_p) | q[w]
                Sgate(eta_p, pt_p * phi_p / 2) | q[w]
        
        # Trainable CX entanglers (same as new_circuit)
        for (a, b) in cx_pairs:
            CXgate(cx_theta[(a, b)]) | (q[a], q[b])
            
        # Additional entanglement layer (same as new_circuit)
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
            
        # Final trainable layer (same as new_circuit)
        for w in range(wires):
            Sgate(squeeze_mag[w], squeeze_phase[w]) | q[w]
            Dgate(disp_mag[w], disp_phase[w]) | q[w]
    
    return prog


def reuploading_circuit(prog, wires, weights, reuploads_per_wire=2):
    """
    Circuit with reuploading encoding — the same particle features are uploaded
    multiple times per wire, but each upload has its own symbolic parameters.

    Notes
    - No interleaved/sequential mapping is required since the source particle per
      wire is the same for each reupload.
    - Expected symbolic parameter naming for the data features per wire and reupload:
        eta_{w}_{r}, pt_{w}_{r}, phi_{w}_{r}  for wire w in [0..wires-1], reupload r in [0..reuploads_per_wire-1]
      Even though both reuploads on a given wire will be fed the same numerical
      particle features at runtime, they use distinct symbols so the graph can
      represent two consecutive encodings.

    Args:
        prog: The Strawberry Fields program object.
        wires: Number of wires in the circuit.
        weights: A dictionary containing circuit parameters. It must include:
            - 's_scale'
            - per-wire Gaussian trainables: squeeze_mag_{w}, squeeze_phase_{w},
              disp_mag_{w}, disp_phase_{w}
            - reupload data symbols per wire: eta_{w}_{r}, pt_{w}_{r}, phi_{w}_{r}
            - optional CX parameters: cx_theta_{a}_{b} for a<b
        reuploads_per_wire: Number of times to reupload the same particle per wire (default 2).
    """

    # global scale parameter
    s_scale = weights['s_scale']

    # per‑mode Gaussian trainables
    squeeze_mag   = [weights[f'squeeze_mag_{i}']   for i in range(wires)]
    squeeze_phase = [weights[f'squeeze_phase_{i}'] for i in range(wires)]
    disp_mag      = [weights[f'disp_mag_{i}']      for i in range(wires)]
    disp_phase    = [weights[f'disp_phase_{i}']    for i in range(wires)]

    # trainable CX coupling strengths (if provided)
    cx_pairs = [(i, j) for i in range(wires) for j in range(i + 1, wires)]
    cx_theta = {(a, b): weights[f"cx_theta_{a}_{b}"] for (a, b) in cx_pairs if f"cx_theta_{a}_{b}" in weights}

    with prog.context as q:
        # data-encoding with reuploads per wire
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        for w in range(wires):
            for r in range(reuploads_per_wire):
                eta_wr = weights[f"eta_{w}_{r}"]
                pt_wr  = weights[f"pt_{w}_{r}"]
                phi_wr = weights[f"phi_{w}_{r}"]

                # Apply encoding gates for this reupload on wire w
                Dgate(scale * pt_wr, eta_wr)      | q[w]
                Sgate(eta_wr, pt_wr * phi_wr / 2) | q[w]

        # (trainable) CX entanglers, if provided
        for (a, b) in cx_pairs:
            key = (a, b)
            if key in cx_theta:
                CXgate(cx_theta[key]) | (q[a], q[b])

        # Additional entanglement layer (same structure as new_circuit)
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])

        # Final per‑mode trainable layer
        for w in range(wires):
            Sgate(squeeze_mag[w], squeeze_phase[w]) | q[w]
            Dgate(disp_mag[w], disp_phase[w])       | q[w]

    return prog

def x8_circuit(prog, wires, weights):
    """
    X8-compatible version of new_circuit FOR SIMULATION (not hardware)!
    This means that we can only work with 4 modes, and the entanglement between the signal and idler modes is assumed. 
    - Sgate instead of Sgate+Dgate pair
    - BSgate instead of CXgate (direct substitution)
    - MZgate to approximate Dgate displacement
    - Rgate instead of final Sgate
    
    Args:
        prog: SF Program 
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
        
        # Data encoding layer - replace Sgate+Dgate pair with Sgate+MZgate
        for w in range(wires):
            r_param = eta[w] 
            phi_param = pt[w] * phi[w] / 2  
            Sgate(r_param, phi_param) | q[w]

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

@operation(4)
def x8_unitary(eta, pt, phi, s_scale, squeeze_mag, squeeze_phase, disp_mag, disp_phase, cx_theta, q):
    """
    X8 hardware-compatible unitary operation for 4 modes.
    Uses only X8-allowed operations: Rgate, MZgate, BSgate.
    
    Args:
        eta, pt, phi: Data encoding parameters (arrays of length 4)
        s_scale: Global scale parameter  
        squeeze_mag, squeeze_phase: Squeeze parameters (arrays of length 4)
        disp_mag, disp_phase: Displacement parameters (arrays of length 4)
        cx_theta: CX coupling parameters (dict with (a,b) keys)
        q: Quantum register (4 modes)
    """
    # Scale calculation (same as original circuit)
    scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
    
    # Data encoding layer - X8 CANNOT use Sgate! Only Rgate/MZgate for data encoding
    for w in range(4):
        # Use Rgate to encode eta and pt*phi as phase rotations
        Rgate(eta[w]) | q[w]  # Encode eta as phase
        Rgate(pt[w] * phi[w] / 2) | q[w]  # Encode pt*phi as phase

        # Use MZgate to encode scale * pt 
        if w < 3:  # Only for first 3 modes to avoid index issues
            phi_internal = eta[w]
            phi_external = scale * pt[w]
            MZgate(phi_internal, phi_external) | (q[w], q[(w+1) % 4])
        
    # Replace CXgate with BSgate (direct substitution - preserves coupling strengths)
    cx_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # All pairs for 4 modes
    for (a, b) in cx_pairs:
        if (a, b) in cx_theta:  # Only apply if parameter exists
            BSgate(cx_theta[(a, b)], 0) | (q[a], q[b])
        
    # Original BSgate entanglement
    for i in range(4):
        idx1 = i
        idx2 = (i + 1) % 4
        BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2])
        
    # Final layer
    for w in range(4):
        # Use Rgate with squeeze_mag and squeeze_phase
        Rgate(squeeze_phase[w] + squeeze_mag[w]) | q[w]  # Combine mag and phase
        
        # Approximate final Dgate with MZgate
        if w < 3:  # Only for first 3 modes to avoid index issues
            phi_internal = disp_phase[w] 
            phi_external = disp_mag[w]
            MZgate(phi_internal, phi_external) | (q[w], q[(w+1) % 4])


def x8_circuit_hardware(prog, wires, weights):
    """
    X8-compatible version of new_circuit DESIGNED FOR HARDWARE!
    Uses custom unitary operation to ensure identical processing on signal and idler modes.
    
    Args:
        prog: SF Program 
        wires: Number of wires 
        weights: Same weight dictionary as new_circuit
    """
    
    wires = 4 # ALWAYS 4 wires for hardware circuit, 0-3 signal modes, duplicated onto 4-7 idler modes
    
    # Extract parameters from weights
    s_scale = weights["s_scale"]
    squeeze_mag   = [weights[f"squeeze_mag_{i}"]   for i in range(wires)]
    squeeze_phase = [weights[f"squeeze_phase_{i}"] for i in range(wires)]
    disp_mag      = [weights[f"disp_mag_{i}"]      for i in range(wires)]
    disp_phase    = [weights[f"disp_phase_{i}"]    for i in range(wires)]
    eta  = [weights[f"eta_{i}"] for i in range(wires)]
    pt   = [weights[f"pt_{i}"]  for i in range(wires)]
    phi  = [weights[f"phi_{i}"] for i in range(wires)]
    
    # CX coupling strengths 
    cx_pairs = [(i, j) for i in range(wires) for j in range(i+1, wires)]
    cx_theta = {(a, b): weights[f"cx_theta_{a}_{b}"] for (a, b) in cx_pairs}

    with prog.context as q:
        # Initial squeezed states between signal and idler modes
        # X8 hardware constraint: only r=1.0 or r=0.0 allowed
        S2gate(1.0) | (q[0], q[4])
        S2gate(1.0) | (q[1], q[5])
        S2gate(1.0) | (q[2], q[6])
        S2gate(1.0) | (q[3], q[7])
        
        # Apply identical unitary to both signal modes (0-3) and idler modes (4-7)
        x8_unitary(eta, pt, phi, s_scale, squeeze_mag, squeeze_phase, 
                   disp_mag, disp_phase, cx_theta) | q[:4]  # Signal modes
        x8_unitary(eta, pt, phi, s_scale, squeeze_mag, squeeze_phase, 
                   disp_mag, disp_phase, cx_theta) | q[4:]  # Idler modes
        
        # Measurement on all modes
        MeasureFock() | q
        
    return prog
