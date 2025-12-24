#### channel_LCU.py ####
 
# The lowering pass of the channel LCU method 
# for Lindbladian simulation from arxiv:1612.09512

#########################

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import UnitaryGate, TGate, RZGate, HGate, C3XGate, StatePreparation

from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector
from channel_IR import *
from copy import deepcopy

### Prepare a superposition state with given coeffs
def prep_sup_state(coeffs: list) -> QuantumCircuit:
    """
    Prepare a superposition state |psi> = sum_j sqrt{c_j} |j>
    where coeffs = [c_0, c_1, ..., c_{M-1}]
    """
    M = len(coeffs)
    norm = np.sqrt(sum([abs(c)**2 for c in coeffs]))
    normalized_coeffs = [c / norm for c in coeffs]
    
    # Create the statevector
    qubit_length = int(np.ceil(np.log2(M)))
    state_vector = np.zeros(2**qubit_length, dtype=complex)
    for j in range(M):
        state_vector[j] = normalized_coeffs[j]

    state_vector = Statevector(state_vector)
    qc = QuantumCircuit(qubit_length)
    qc.initialize(state_vector)
    # qc = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=3)
    return qc

### Construct a single channel from Lindbladian
def Lindblad_to_channel(Lind: Lindbladian, delta_t: float) -> channel_ensemble:
    sqdelta = np.sqrt(delta_t)
    channels = []
    
    for Ls in Lind.L_list:
        
        new_coeff_list = [] 
        new_mat = []
        for ms in Ls.mat_list: 
            coeff = ms.coeff * sqdelta
            new_mat.append(ms.mat)
            new_coeff_list.append(coeff)

        kraus_op_ls = Matrixsum(new_mat, new_coeff_list)

        channels.append(kraus_op_ls)

    c = channel_ensemble(channels)
    return c 

def multiplexed_U(ctrl_qubits: QuantumRegister, select_qubits: QuantumRegister, sys_qubits: QuantumRegister, mats: list):
    
    ## Size of unitaries 
    rows = len(mats)
    cols = len(mats[0]) if rows > 0 else 0
    
    qubit_size_tot = len(ctrl_qubits) + len(select_qubits) + len(sys_qubits)
    qc_comp = QuantumCircuit(ctrl_qubits, select_qubits, sys_qubits)

    ## Generating multiplexed U by iterating over control values. 
    for i in range(rows):
        for j in range(cols):
            # qc_elem = QuantumCircuit(qubit_size_tot)
            
            ## If mats[i][j] less than that of sys qubits, need to pad identity.
            if mats[i][j].shape[0] < 2**len(sys_qubits):
                pad_size = 2**len(sys_qubits) // mats[i][j].shape[0]
                mats[i][j] = np.kron(mats[i][j], np.eye(pad_size))
            
            U_elem = UnitaryGate(mats[i][j])
            control_values =  bin(i)[2:].zfill(len(select_qubits)) + bin(j)[2:].zfill(len(ctrl_qubits)) 
            
            ctrl_U_elem = U_elem.control(num_ctrl_qubits = len(ctrl_qubits) + len(select_qubits), ctrl_state = control_values)
            qc_comp.append(ctrl_U_elem, range(qubit_size_tot))
            
    return qc_comp

def multiplexed_B(ctrl_size, select_size: int, coeffs: list):
    """
    Prepare the multi-B operation: multi-B |0>|j> = 1/sqrt(s_j) sum_k sqrt(a_{jk}) |k>|j>
    where coeffs = [[a_{00}, a_{01}, ...], [a_{10}, a_{11}, ...], ...]
    """
    rows = len(coeffs)
    cols = len(coeffs[0]) if rows > 0 else 0

    qubit_size_tot = ctrl_size + select_size
    qc_B = QuantumCircuit(qubit_size_tot)

    for i in range(rows):
        ### Prepare k subroutines: B_sub_j |0> = 1/sqrt(s_j) sum_k sqrt(a_{jk}) |k>
        ### coeffs_i: [a_{ik}]
        coeffs_i = coeffs[i]
        
        norm_i = np.sqrt(sum([abs(c) for c in coeffs_i]))
        
        normalized_coeffs_i = [np.sqrt(c) / norm_i for c in coeffs_i]
        
        state_vector = np.zeros(2**ctrl_size, dtype=complex)
        for j in range(cols):
            state_vector[j] = normalized_coeffs_i[j]
        
        state_vector = Statevector(state_vector)
        
        qc_temp = QuantumCircuit(ctrl_size)
        
        qc_temp.append(StatePreparation(state_vector), range(ctrl_size))
        control_values = bin(i)[2:].zfill(select_size)
        ctrl_qc_temp = qc_temp.to_gate().control(num_ctrl_qubits=select_size, ctrl_state=control_values)
        sel_index = list(range(ctrl_size, ctrl_size + select_size))
        ctrl_index = list(range(ctrl_size))
        qc_B.append(ctrl_qc_temp, sel_index + ctrl_index)
        
    # qc_B = transpile(qc_B, basis_gates=['x', 'y', 'z', 'rx', 'ry', 'h', 'cx'], optimization_level=1)
    return qc_B


def channel_to_LCU (ensem: channel_ensemble) -> list:
    """
    Convert the channel to LCU circuit, only allow single channel for this method.
    
    Method: lCU W|0>|mu>|psi> = sqrt(p)|0> sum_j |j> A_j |psi> + sqrt(1-p)|1>|Phi^perp>
    W = multi-B^{dag} multi-U multi-B 
    multi-B |0> |j> = 1/sqrt(s_j) sum_k sqrt(a_{jk}) |k>|j>
    multi-U |k>|j>|psi> = |k>|j> U_{jk} |psi>

    |mu> = 1/sqrt(sum_j s_j^2) sum_j s_j |j>,
    s_j = sum_k a_{jk}
    Arguments: 
    ensem: length-1 channel_ensemble, channel: the only one channel

    sys_size: size of |psi> the system
    select_size: size of |j> the select register
    ctrl_size : size of |0> the control qubit
    """
    assert len(ensem.channels) == 1
    
    channel = ensem.channels[0][1]

    sys_size = channel[0].mat_list[0].mat.shape[0].bit_length() - 1
    ## 2* select size is the count of Kraus operators (j)
    select_size = int(np.ceil(np.log2(len(channel))))
    ### 2**ctrl size is the count of Pauli terms (k) in each Kraus operator.  k indexes before j.
    ctrl_size = len(channel[0].mat_list).bit_length() - 1
    ## Store elementary matrices and their coefficients. 
    coeff_channel = [
        [0 for _ in range(len(channel[0].mat_list))]
        for _ in range(len(channel))
    ]
    mats_channel = [
        [0 for _ in range(len(channel[0].mat_list))]
        for _ in range(len(channel))
    ]
   
    coeff_sums = []
    for i, ms in enumerate(channel):
        
        coeff_ms_sum = 0
        
        assert isinstance(ms, Matrixsum)
        for j, mat in enumerate(ms.mat_list):
            coeff_channel[i][j] = abs(mat.coeff)
            coeff_ms_sum += abs(mat.coeff)
            mats_channel[i][j] = mat.mat

        coeff_sums.append(coeff_ms_sum)
    ## Prepare the superposition state 
    qc = QuantumCircuit(sys_size + select_size + ctrl_size)
    LCU_ini = prep_sup_state(coeff_sums)
    ### Padding initialization
    qc.compose(LCU_ini, qubits=list(range(ctrl_size, ctrl_size + select_size)), inplace=True) 
    ### 
    multiplex_B_circuit = multiplexed_B(ctrl_size, select_size, coeff_channel)

    qc.compose(multiplex_B_circuit, qubits = list(range(select_size + ctrl_size)), inplace = True)
    # qc.save_statevector('multi-B') #type: ignore 
    # if trunc > 1:
    qc.compose(multiplexed_U(
        ctrl_qubits = qc.qubits[0:ctrl_size],
        select_qubits = qc.qubits[ctrl_size: ctrl_size + select_size],
        sys_qubits = qc.qubits[ctrl_size + select_size: ],
        mats = mats_channel
    ), qubits = qc.qubits, inplace=True)
        # qc.save_statevector('multi-U') #type:ignore
    # if trunc > 2:
    qc.compose(multiplex_B_circuit.inverse(), qubits = list(range(select_size + ctrl_size)), inplace = True)
    qc.save_statevector('final_state') #type: ignore
    return [qc, LCU_ini]


def oblivious_AA(ctrl_size, select_size, sys_size, circuit: QuantumCircuit) -> QuantumCircuit:
    """
    To be continued: Oblivious amplitude amplification for channel LCU
    """
    qc = QuantumCircuit(ctrl_size + select_size + sys_size)
    return qc


def channel_to_LCU_rand (ensem: channel_ensemble) -> QuantumCircuit:
    """
    Convert the channel to LCU circuit with random sampling method
    """
    ### To be continued 
    assert len(ensem.channels) > 1
    channels = ensem.channels
    probs = [ch[0] for ch in channels]
    channel_ops = [ch[1] for ch in channels]
    qc = QuantumCircuit(1)
    return qc

def approx_state(sv, tol: float = 1e-8):
    """
    Approximate a vector/matrix by removing small amplitudes
    """
    sv_array = sv.data.copy()
    sv_array.real[np.abs(sv_array.real) < tol] = 0.
    sv_array.imag[np.abs(sv_array.imag) < tol] = 0. 
    return Statevector(sv_array)
def test_circuit(circuit: QuantumCircuit, ini_circuit: QuantumCircuit) -> float:
    N = 2000
    simulator = AerSimulator()
    qc_test = deepcopy(circuit)

    # ini_state = Statevector.from_label('0' * circuit.num_qubits)
    ini_state = Statevector.from_label('0000')
    # print(qc_test.draw())
    qc_test = transpile(qc_test, simulator, optimization_level=3)
    
    ## Small example, use statevector test 
    result_job = simulator.run(qc_test, shots = 1, initial_state=ini_state).result().data(0)

    # start_sv = result_job['multi-B']
    # start_sv = approx_state(Statevector(start_sv), tol=1e-6)
    # start_state = start_sv.to_dict()
    # print("Start statevector:", start_state)

    # middle_sv = result_job['multi-U']
    # middle_sv = approx_state(Statevector(middle_sv), tol=1e-6)
    # middle_state = middle_sv.to_dict()
    # print("Middle statevector:", middle_state)

    final_sv = result_job['final_state']
    final_sv = approx_state(Statevector(final_sv), tol=1e-6).data
    final_state = Statevector(final_sv).to_dict()
    print("Final statevector:", final_state) 

    creg = ClassicalRegister(1, 'c')
    qc_test.add_register(creg)
    qc_test.measure(0, creg[0])
    
    result = simulator.run(qc_test, shots = N, initial_state=ini_state).result()
    success_prob = result.get_counts()['0'] / N
    # print("Success probability:", success_prob)

    # qc_test.save_statevector() #type: ignore
    # final_sv = simulator.run(qc_test, shots = 1, initial_state=ini_state).result().data(0)['statevector'].data
    # final_state = Statevector(final_sv)

    # return final_state
    return success_prob

if __name__ == "__main__":
    A = [(['YI', 'IY'], [1, 0.25]), (['XI','IX'], [0.5, 0.75])]
    channel = []

    for i in range(len(A)):
        channel.append(Matrixsum(A[i][0], A[i][1]))
    singleton = channel_ensemble([channel])
    # ctrl_size = select_size = 1
    # sys_size = 2
    
    LCU, ini = channel_to_LCU(singleton)
    success_prob = test_circuit(LCU, ini)
    print(success_prob)

    # final_state = test_circuit(LCU, ini)
    qreg = QuantumRegister(2, 'q')
    creg = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qreg, creg)
    qc.h(qreg[0])
    qc.cx(qreg[0], qreg[1])
    qc.measure(qreg[0], creg[0])
    simulator = AerSimulator()
    result = simulator.run(qc, shots = 100000).result()
    counts = result.get_counts()['0']
    # print(counts)
    # ini_state_ground = Statevector(qc_ini)
    
   


  