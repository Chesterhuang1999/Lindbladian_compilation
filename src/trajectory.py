##### trajectory.py ##### 

# Simulate the lindbladian dynamics 
# via quantum trajectory method, 
# i.e. randomly picks a channel from the ensemble 

#########################

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import UnitaryGate, TGate, RZGate, HGate, C3XGate, StatePreparation

from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector
from channel_IR import *
from channel_LCU import simulate_circuit, approx_state
from copy import deepcopy

def lindblad_to_rand_channels(Lind: Lindbladian, delta_t: float) :
    """
    Convert the Lindbladian to channel ensemble representation
    via the Kraus operator expansion
    H = sum_i T_0l V_0l, L_j = sum_l T_jl V_jl
    """
    H = Lind.H
    L_list = Lind.L_list
    channels = []
    probs = []
    lind_pauli_norm = Lind.pauli_norm()

    iden = PauliAtom('I' * int(np.log2(H.instances[0][0].bare_op().dim[0])), phase = 1.0)
    ## Define F-channels 
    for H_insts in H.instances: 
        mats, coeff = H_insts
        new_mats = PauliAtom(mats.expr, phase = mats.phase * -1j)

        kraus_H_op = Matrixsum([(iden, 1.0), (new_mats, delta_t * coeff * lind_pauli_norm)])
        channel_inst = [kraus_H_op]
        channels.append(channel_inst)
        probs.append(coeff / lind_pauli_norm)
    for L in L_list:
        kraus_L_op = []
        L_cpy = deepcopy(L)
        coeff_L = L_cpy.pauli_norm()
        L_prod = matsum_mul(L_cpy.adj(), L_cpy)
        L_prod.mul_coeffs(-0.5 * delta_t * lind_pauli_norm / (coeff_L**2))
        L_prod.instances.append((iden, 1.0))
        kraus_L_op.append(L_prod)
        L_cpy.mul_coeffs(np.sqrt(delta_t * lind_pauli_norm) / coeff_L)
        kraus_L_op.append(L_cpy)
        probs.append((coeff_L**2) / lind_pauli_norm)
        channels.append(kraus_L_op)
    return channel_ensemble(channels, probs), len(H.instances)

#### ToDo: Construct the LCU circuit for a given channel (an improved LCU design for channels)
def channel_to_LCU_det(ops, pauli_norm: float, delta_t: float, is_Ham: bool, size: tuple) -> QuantumCircuit:
    """ Design the quantum circuit with regard to the channel picked"""
    control_size, sys_size = size
    if is_Ham == True:
        ## First order poly LCU for Hamiltonian
        qc = first_order_poly_LCU(ops, pauli_norm, delta_t, sys_size)
    else: 
        ## Second order poly LCU for Lindbladian 
        qc = second_order_poly_LCU(ops, pauli_norm, delta_t, size)
    return qc

### Subroutine: first-order polynomial for LCU
def first_order_poly_LCU(ops, pauli_norm: float, delta_t: float, sys_size: int) :
    pauli_op, coeff = ops
    qc = QuantumCircuit(2 + sys_size)
    alpha = np.arctan(delta_t * pauli_norm)
    p1 = 1/(1 + pauli_norm * delta_t)**2 
    p = 1 - 2 * pauli_norm * delta_t 
    
    beta = np.arctan(np.sqrt(p1/p) - 1) /2 
    theta1 = np.arccos(np.sqrt(np.cos(alpha)) / np.sqrt(np.cos(alpha) + np.sin(alpha)))
    theta2 = np.arccos(np.sqrt(np.cos(beta)) / np.sqrt(np.cos(beta) + np.sin(beta)))
    
    qc.z(0)
    qc.ry(2 * theta1 , 0)
    qc.z(1)
    qc.ry(2 * theta2 , 1)
    
    qc_pauli = QuantumCircuit(sys_size)
    qc_pauli.append(Pauli(pauli_op.expr), range(sys_size))
    qc_pauli.global_phase = np.angle(pauli_op.phase * -1j)
    qc_pauli = qc_pauli.decompose()
    U_elem_ctrl = qc_pauli.to_gate().control(1, ctrl_state = '1')
    qc.append(U_elem_ctrl, [0] + list(range(2, 2 + sys_size)))
    
    ### Rotation again
    qc.z(0)
    qc.ry(2 * theta1, 0)
    qc.z(1)
    qc.ry(2 * theta2, 1)
    return qc

def initialize_state(coeffs: list, ctrl_reg_size: int) -> QuantumCircuit:
    norm = np.sqrt(sum(coeffs))
    state_vec = np.zeros(2**ctrl_reg_size, dtype = float)
    for j in range(len(coeffs)):
        state_vec[j] = np.sqrt(coeffs[j]) / norm
    assert 2**ctrl_reg_size >= len(coeffs)
    qc = QuantumCircuit(ctrl_reg_size)
    state_vec = Statevector(state_vec)
    qc.initialize(state_vec, range(ctrl_reg_size))
    qc = qc.decompose()
    for i, instruction in enumerate(qc.data):
        if instruction.operation.name == 'reset':
            del qc.data[i]
            break
    
    qc_transpile = transpile(qc, basis_gates = ['ry', 'rz', 'rx', 'cx'], optimization_level=3)

    return qc_transpile
def controlled_paulis(paulis: list, ctrl_size: int, sys_size: int):
    qc = QuantumCircuit(ctrl_size + sys_size)
    for j in range(len(paulis)):
        pauli_op = paulis[j]
        qc_pauli = QuantumCircuit(sys_size)
        qc_pauli.append(Pauli(pauli_op.expr), range(sys_size))
        qc_pauli.global_phase = np.angle(pauli_op.phase)
        qc_pauli = qc_pauli.decompose()
        ctrl_value = bin(j)[2:].zfill(ctrl_size)
        U_elem_ctrl = qc_pauli.to_gate().control(ctrl_size, ctrl_state = ctrl_value)
        qc.append(U_elem_ctrl, list(range(ctrl_size + sys_size)))
    qc_transpile = transpile(qc, basis_gates = ['ry', 'rz', 'rx', 'cx'], optimization_level=3)
    return qc

def second_order_poly_LCU(ops, pauli_norm: float, delta_t: float, size: tuple):
    unit = pauli_norm * delta_t / 2
    p2 = 1 / (2 * unit + (1 + unit)**2)
    p = 1 - 4 * unit
    beta = np.arctan(np.sqrt(p2 / p) - 1) / 2
    control_size, sys_size = size
    paulis = [ops.instances[j][0] for j in range(len(ops.instances))]
    coeffs = [ops.instances[j][1] for j in range(len(ops.instances))]
    # Similar with channel_LCU: prepare control registers with coeffs, and prepare LCU with paulis
    qc = QuantumCircuit(control_size + sys_size + 1)
    ctrl_reg_size = int((control_size - 1)/ 2 )
    ## Prepare control registers 
    qc_ini = initialize_state(coeffs, ctrl_reg_size)
    qc.compose(qc_ini, qubits = list(range(3, 3 + ctrl_reg_size)), inplace = True)
    qc.compose(qc_ini, qubits = list(range(3 + ctrl_reg_size, 3 + 2 * ctrl_reg_size)), inplace = True)
    alpha1 = np.arctan(2 * unit / (1 + unit)**2)
    alpha2 = np.arctan(unit)
    theta1 = np.arccos(np.sqrt(np.cos(alpha1)) / np.sqrt(np.cos(alpha1) + np.sin(alpha1)))
    theta2 = np.arccos(np.sqrt(np.cos(alpha2)) / np.sqrt(np.cos(alpha2) + np.sin(alpha2)))
    theta3 = np.arccos(np.sqrt(np.cos(beta)) / np.sqrt(np.cos(beta) + np.sin(beta)))

    qc.z(0)
    qc.ry(2 * theta1 , 0)
    qc.z(1) 
    qc.ry(2 * theta2 , 1)
    qc.z(2)
    qc.ry(2 * theta3 , 2)
    
    ### Apply controlled LCU for Paulis 
    paulis_neg = []
    paulis_adj = []
    for p in paulis:
        p_cpy1 = deepcopy(p)
        p_cpy1.phase = - p_cpy1.phase 
        paulis_neg.append(p_cpy1)
        p_cpy2 = deepcopy(p)
        p_cpy2 = p_cpy2.adjoint()
        paulis_adj.append(p_cpy2)
    
    multiplexed_pauli_1 = controlled_paulis(paulis, ctrl_reg_size, sys_size).control(1, ctrl_state = '1')
    multiplexed_pauli_2 = controlled_paulis(paulis_neg, ctrl_reg_size, sys_size).control(2, ctrl_state = '10')
    multiplexed_pauli_3 = controlled_paulis(paulis_adj, ctrl_reg_size, sys_size).control(2, ctrl_state = '10')
    qc.append(multiplexed_pauli_1, [0] + list(range(3 + ctrl_reg_size, 3 + 2 * ctrl_reg_size + sys_size)))
    qc.append(multiplexed_pauli_2, [0, 1] + list(range(3 + ctrl_reg_size, 3 + 2 * ctrl_reg_size + sys_size)))
    qc.append(multiplexed_pauli_3, [0, 1] + list(range(3, 3 + ctrl_reg_size)) + list(range(3 + 2 * ctrl_reg_size, 3 + 2 * ctrl_reg_size + sys_size)))
    ### Reverse rotation

    qc.z(1)
    qc.ry(2 * theta2, 1)
    qc.z(2)
    qc.ry(2 * theta3, 2)
    
    qc_ini_inv = qc_ini.inverse()
    qc.compose(qc_ini_inv, qubits = list(range(3 + ctrl_reg_size, 3 + 2 * ctrl_reg_size)), inplace = True)
    qc.compose(qc_ini_inv, qubits = list(range(3, 3 + ctrl_reg_size)), inplace = True)
    return qc

## Oblivious Amplitude Amplification
### Here another ancilla reg is used to modulate the success prob to 1/2 manually
###  so that oblivious AA can boost the success prob to unity
def reflection_op(qubit_size: int, proj_size: list):
    ### Construct a reflection operator (I - 2|0><0|) otimes I over qubit_size
    qc = QuantumCircuit(qubit_size)
    qc.x(proj_size)
    qc.h(proj_size[-1])
    if qubit_size > 1:
        qc.mcx(proj_size[:-1], proj_size[-1])
    qc.h(proj_size[-1])
    qc.x(proj_size)
    return qc
def oblivious_AA(qc_main:QuantumCircuit, qc_sub, qubit_regs):
    """ 
    Implement the oblivious amplitude amplification 
    qc_main: the main quantum circuit
    qc_sub: the subcircuit to be amplified and embedded
    qubit_regs: indexes of the qubit registers.
    """
    
    qc_sub_inv = qc_sub.inverse()
    sel_size, ctrl_size, sys_size, anc_size = qubit_regs
    
    ### Generate reflection operators
    total_size = sel_size + ctrl_size + sys_size + anc_size

    proj_size1 = list(range(sel_size, sel_size + ctrl_size)) + [total_size -1]
    qc_refl_A = reflection_op(total_size, proj_size1)

    proj_size2 = list(range(sel_size + ctrl_size)) + [total_size -1]
    qc_refl_B = reflection_op(total_size, proj_size2)

    ### Construct the amplified circuit  
    qc_main.compose(qc_sub, inplace = True)
    qc_main.compose(qc_refl_A, inplace = True)
    qc_main.compose(qc_sub_inv, inplace = True)
    qc_main.compose(qc_refl_B, inplace = True)
    qc_main.compose(qc_sub, inplace = True)
    qc_main.global_phase += np.pi
    return qc_main

def channel_to_trajectory_rand (Lind, time: float, epsilon: float) -> QuantumCircuit:
    """
    Convert the channel to LCU circuit with random sampling method
    """
    ### To be continued 
    pauli_norm = Lind.pauli_norm()
    tau = int(np.ceil(2* pauli_norm * time))
    r = int(np.pow(2, np.ceil(np.log2(tau / epsilon))))
    
    delta_t = time / (r * tau)
    p = 1 - 2 * pauli_norm * delta_t
    ensem , H_length = lindblad_to_rand_channels(Lind, delta_t)
    print(tau, r, delta_t, p)
    print(tau/epsilon)
    exit(0)
    assert len(ensem.channels) > 1
    channels = ensem.channels
    probs = [ch[0] for ch in channels]
    # channel_ops = [ch[1] for ch in channels]
    
    ## Set parameters 
   
    sys_size = ensem.size 
    r = 2 
    select_reg = QuantumRegister(r, 'sel')
    control_size = 2 * int(np.ceil(np.log(ensem.length[0]))) + 2 
    control_reg = QuantumRegister(control_size * r, 'ctrl')
    sys_reg = QuantumRegister(sys_size, 'sys')
    anc_reg = QuantumRegister(tau, 'anc')

    preset_prob = 1 - 2 * pauli_norm * delta_t
    amp = np.pow(preset_prob, -r/ 2) /2 
    
    rot_angle = - 2 * np.arccos(amp)
    
    sub_circ_size = (control_size, sys_size)
    
    ## select_reg: 0- r-1, control_reg: (r - r + ctrl_size * (r - 1), sys_size: last qubits
    
    QC = QuantumCircuit(select_reg, control_reg, sys_reg, anc_reg)
    for k1 in range(tau):
        qc_main = QuantumCircuit(select_reg, control_reg, sys_reg, anc_reg)
        for k2 in range(r):
            rand_ind = np.random.choice(len(channels), p = probs)
            rand_ind = 1 + H_length # For testing purpose
            is_Ham = True if rand_ind < H_length else False
            if is_Ham: 
                selected_ops = Lind.H.instances[rand_ind] ## tuple(PauliAtom, coeff)
            else:
                selected_ops = Lind.L_list[rand_ind - H_length] ## Matrixsum
            
            
            ini_state = Statevector.from_label('0' * sys_size)
            
            sub_qc = channel_to_LCU_det(selected_ops, pauli_norm, delta_t, is_Ham, sub_circ_size)
            
            temp_value1 = r + control_size * k2
            temp_value2 = (control_size + 1) * r
            ## Embedd the selected channel
            if is_Ham: 
                qc_main.compose(sub_qc, qubits = [temp_value1, temp_value1 + 1] + list(range(temp_value2, temp_value2 + sys_size)), inplace = True)
            else:
                qc_main.compose(sub_qc, qubits = [k2] + list(range(temp_value1, temp_value1 + control_size)) + 
                    list(range(temp_value2, temp_value2 + sys_size)), inplace = True)
        qc_main.ry(rot_angle, anc_reg[k1])
        
        QC = oblivious_AA(QC, qc_main, [r, control_size * r, sys_size, 1])
    return QC


def simulate(circuit: QuantumCircuit, ini_state: Statevector):
    """ Simulate the quantum circuit with initial state """
    N = 20000
    simulator = AerSimulator()
    qc_test = deepcopy(circuit)
    
    qc_test = transpile(qc_test, simulator, optimization_level=1)
    print(qc_test.draw())
    qc_test.save_statevector('final_state') #type: ignore

    
    result_job = simulator.run(qc_test, shots = 1, initial_state=ini_state).result().data(0)

    final_sv = result_job['final_state']
    final_sv = approx_state(Statevector(final_sv), tol=1e-6)
    creg = ClassicalRegister(2, 'c')
    qc_test.add_register(creg)
    for i in range(2):
        qc_test.measure(i, creg[i])
    result = simulator.run(qc_test, shots = N, initial_state=ini_state).result()
    success_prob = result.get_counts()['00'] / N
    return success_prob, final_sv

if __name__ == "__main__":
    # H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    # gamma = np.sqrt(0.1)/2 
    # L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    # delta_t = 0.1
    # TFIM_lind = Lindbladian(H, L_list)

    # TFIM_qc_exe = channel_to_trajectory_rand(TFIM_lind, 0.1, 0.01)
    H = [('I', 1)]
    v = 0.5
    t = 0.1
    K1 = 5
    L_list = [[('X', np.sqrt(v + 1)), ('Y', 1j * np.sqrt(v + 1))], [('X', np.sqrt(v)), ('Y', -1j * np.sqrt(v))]]
    decay_lind = Lindbladian(H, L_list)
    # print(TFIM_qc_exe.draw())
    decay_qc_exe = channel_to_trajectory_rand(decay_lind, t, 0.04)
    ini_state = Statevector.from_label('0' * (decay_qc_exe.num_qubits))
    success_prob, final_sv = simulate(decay_qc_exe, ini_state)
    print(success_prob)
    print(final_sv)
    # print(final_sv / success_prob)
    