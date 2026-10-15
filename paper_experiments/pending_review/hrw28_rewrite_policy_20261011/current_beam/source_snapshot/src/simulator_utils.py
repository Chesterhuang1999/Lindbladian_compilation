from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, Operator, partial_trace
import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.library import SetDensityMatrix

def zero_small_complex(arr, tol=1e-8):
    """
    If |Re| < tol, set Re = 0;
    If |Im| < tol, set Im = 0.
    """
    arr = arr.copy()
    re = arr.real
    im = arr.imag

    re[np.abs(re) < tol] = 0.0
    im[np.abs(im) < tol] = 0.0

    return re + 1j * im
def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
def projection_vec(sv: Statevector, dim: int, anc_size: int, ctrls: list):
    sel_size = anc_size - len(ctrls)
    
    proj_0 = Operator.from_label('0' * len(ctrls))
    if sel_size == 0:
        iden = Operator.from_label('I' * (dim - anc_size))
        proj_full = iden.tensor(proj_0)
        projected_vec = np.dot(proj_full, sv)
        projected_vec = projected_vec[::2**anc_size] 
        projected_vec = normalize(projected_vec)
        return Statevector(zero_small_complex(projected_vec, tol = 1e-6))
    else:
        iden_sel = Operator.from_label('I' * (sel_size))
        iden_sys = Operator.from_label('I' * (dim - anc_size))
        proj_full = iden_sys.tensor(proj_0).tensor(iden_sel) 
        projected_vec = np.dot(proj_full, sv)
        projected_dm_sys = partial_trace(DensityMatrix(projected_vec), list(range(anc_size)))
        # projected_dm_sys = partial_trace(DensityMatrix(projected_vec), list(range(sel_size, anc_size)))
        return DensityMatrix(zero_small_complex(np.asarray(projected_dm_sys), tol = 1e-6))
    

def projection_op_dm(dm: DensityMatrix, reg_sizes: list):
    sel_size, ctrl_size, sys_size = reg_sizes
    proj_0 = Operator.from_label('0' * ctrl_size)
    iden_sel = Operator.from_label('I' * sel_size)
    iden_sys = Operator.from_label('I' * sys_size)
    proj_full = iden_sys.tensor(proj_0).tensor(iden_sel)
    projected_dm = np.dot(np.dot(proj_full, dm), proj_full.adjoint())
    
    projected_dm_sys = partial_trace(DensityMatrix(projected_dm), list(range(sel_size + ctrl_size)))
    return DensityMatrix(zero_small_complex(np.asarray(projected_dm_sys), tol = 1e-6))

def simulate_circuit_statevec(qc: QuantumCircuit, ini_state, sel_state, reg_sizes: list):
    simulator = AerSimulator(method = 'statevector')
    
    sel_size, ctrl_size, sys_size = reg_sizes
    qc_sim = QuantumCircuit(qc.num_qubits)
    if isinstance(ini_state, str):
    
        state_sys_ctrl = Statevector.from_label(ini_state + '0' * ctrl_size)
    else:
        state_sys_ctrl = ini_state
    if sel_size > 0 and sel_state is not None: 
        state_tot = state_sys_ctrl.tensor(Statevector(sel_state))
    else:
        state_tot = state_sys_ctrl
    
    qc_sim.initialize(state_tot)

    qc_sim.compose(qc, qc_sim.qubits, inplace=True)
    qc_sim.save_statevector(label = 'final_state') #type: ignore
    qc_sim = transpile(qc_sim, simulator, optimization_level=2)

    ### Sim task I: Get final statevector
    
    result = simulator.run(qc_sim, shots = 1).result()
    final_state = result.data()['final_state']
    
    final_state_sys = projection_vec(final_state, sel_size + ctrl_size + sys_size, sel_size + ctrl_size, list(range(sel_size, sel_size + ctrl_size)))
    return final_state_sys
    # return final_dm

def simulate_circuit_dm(qc: QuantumCircuit, ini_state, sel_state, reg_sizes: list):
    simulator = AerSimulator(method = 'density_matrix')
    
    sel_size, ctrl_size, sys_size = reg_sizes
    qc_sim = QuantumCircuit(qc.num_qubits)

    dens_sys_ctrl = ini_state.tensor(DensityMatrix.from_label('0' * ctrl_size))
    dens_tot = dens_sys_ctrl.tensor(DensityMatrix(Statevector(sel_state)))
    # qc = transpile(qc, simulator, optimization_level=1)
    qc_sim.append(SetDensityMatrix(dens_tot), qc_sim.qubits)
    qc_sim.compose(qc, qc_sim.qubits, inplace=True)
    # qc_sim.append(qc.to_gate(), qargs = qc_sim.qubits)
    qc_sim.save_density_matrix(label = 'final_dm') #type: ignore
    qc_sim = transpile(qc_sim, simulator, optimization_level=1)
    print("Transpile complete, starting simulation... ")
    result = simulator.run(qc_sim, shots = 1).result()
    print("Simulation complete, processing results... ")
    final_dm = result.data()['final_dm']

    final_dm_sys = projection_op_dm(final_dm, [sel_size, ctrl_size, sys_size])
    
    
    return final_dm_sys