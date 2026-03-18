#### channel_LCU.py ####
 
# The lowering pass of the channel LCU method 
# for Lindbladian simulation from arxiv:1612.09512

#########################

import numpy as np
from copy import deepcopy

from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace
from qiskit.circuit.library import UnitaryGate, TGate, RZGate, HGate, C3XGate, StatePreparation
from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector
from subroutine import *
from channel_IR import *
from block_encoding import BlockEncoding
from baseline import simulate_lindblad
from qutip import tensor, Qobj, basis

### Prepare a superposition state with given coeffs


### operator norm for Lindbladian channel (check approximity to identity)
def Lindblad_opnorm(Lind: Lindbladian, delta_t: float) -> float:
    """
    Compute the norm for Lindbladian channel M_delta
    """
    iden = PauliAtom('I' * int(np.log2(Lind.H.instances[0][0].to_operator().dim[0])), phase = 1.0)
    kraus_0_basis = Matrixsum([(iden, 1.0)])
    H_copy = deepcopy(Lind.H)
    H_square = matsum_mul(H_copy, H_copy)
    H_square.mul_coeffs(delta_t**2)
    kraus_0_basis = kraus_0_basis.add(H_square)
    sum_Ls_product = Matrixsum([])
    for Ls in Lind.L_list:
        Ls_copy = deepcopy(Ls)
        Ls_product = matsum_mul(Ls_copy.adj(), Ls_copy)
        sum_Ls_product = sum_Ls_product.add(Ls_product)
    
    Ls_square = matsum_mul(sum_Ls_product, sum_Ls_product)
    
    Ls_square.mul_coeffs(delta_t**2/4)
    kraus_0_basis = kraus_0_basis.add(Ls_square)
    norm = kraus_0_basis.operator_norm()
    return norm

def Lindblad_paulinorm(Lind: Lindbladian) -> float:
    """
    Compute the Pauli norm for Lindbladian channel M_delta
    """
    H_norm = Lind.H.pauli_norm()
    sum_Ls_norm = 0.0 
    for Ls in Lind.L_list:
        Ls_copy = deepcopy(Ls)
        sum_Ls_norm += Ls_copy.pauli_norm()**2
    norm = H_norm + sum_Ls_norm
    return norm

def channel_norm_zero(channel: list, psi: Statevector) -> float:
    """ Compute the norm of a channel at input state |psi><psi|"""
    norm = 0.0
    for Ls in channel: 
        Ls_copy = deepcopy(Ls)
        Ls_product = matsum_mul(Ls_copy.adj(), Ls_copy)
        Ls_op = Ls_product.eff_op()
        
        Ls_psi = psi.evolve(Operator(Ls_op)) # type: ignore
        norm += psi.inner(Ls_psi).real

    return norm


### Construct a single channel from Lindbladian, p
def Lindblad_to_channel(Lind: Lindbladian, delta_t: float):
    sqdelta = np.sqrt(delta_t)
    channels = []
    size = Lind.H.size if Lind.H.size != 0  else Lind.L_list[0].size
    # iden = PauliAtom('I' * int(np.log2(Lind.H.instances[0][0].to_operator().dim[0])), phase = 1.0)
    iden = PauliAtom('I' * size, phase = 1.0)
    kraus_0_basis = Matrixsum([(iden, 1.0)])

    H_copy = deepcopy(Lind.H)
    H_copy.mul_coeffs(-1j* delta_t)
    kraus_0_basis = kraus_0_basis.add(H_copy)
    Ls_copy = []
    for Ls in Lind.L_list:
        Ls_copy.append(deepcopy(Ls))
    for i in range(len(Ls_copy)):
        
        Ls_product = matsum_mul(Ls_copy[i].adj(), Ls_copy[i])
        
        Ls_product.mul_coeffs(-0.5 * delta_t)
        
        kraus_0_basis = kraus_0_basis.add(Ls_product)
    
    channels.append(kraus_0_basis)
         
    for Ls in Lind.L_list:
        Ls_copy_2 = deepcopy(Ls)
        
        Ls_copy_2.mul_coeffs(sqdelta)
        
        channels.append(Ls_copy_2)

    ### Calculate the success prob here
    coeff_sq = 0.0
    coeff_ms_sum = []
    for matsums in channels:
        coeff_sum = matsums.pauli_norm() 
        coeff_sq += coeff_sum**2
        coeff_ms_sum.append(coeff_sum)
    success_prob_th = 1 / coeff_sq
    c = channel_ensemble([channels])
    return c, success_prob_th, coeff_ms_sum

### The multiplexed U operation

def multiplexed_U(ctrl_size: int, select_size: int, sys_size: int, mats: list):
    
    ## Size of unitaries 
    rows = len(mats)
    qubit_size_tot = ctrl_size + select_size + sys_size
    
    qc_comp = QuantumCircuit(qubit_size_tot)
    ## Generating multiplexed U by iterating over control values. 
    for i in range(rows):
        for j in range(len(mats[i])):
            ## If mats[i][j] less than that of sys qubits, need to pad identity.
            if isinstance(mats[i][j], SparsePauliOp):

                pauli_op, phase = mats[i][j].paulis[0], mats[i][j].coeffs[0]
                qc_pauli = QuantumCircuit(pauli_op.num_qubits)
                qc_pauli.append(pauli_op, range(mats[i][j].num_qubits))
                qc_pauli.global_phase = np.angle(phase)
                qc_pauli = qc_pauli.decompose()
                U_elem = qc_pauli.to_gate()
            else:
                if mats[i][j].shape[0] < 2**sys_size:
                    pad_size = 2**sys_size // mats[i][j].shape[0]
                    mats[i][j] = np.kron(mats[i][j], np.eye(pad_size))
                U_elem = UnitaryGate(mats[i][j])

            control_values =  bin(i)[2:].zfill(select_size) + bin(j)[2:].zfill(ctrl_size) 
            if len(qc_pauli.data) > 0: ### Identity is ignored 
                ctrl_U_elem = U_elem.control(num_ctrl_qubits = ctrl_size + select_size, ctrl_state = control_values)
                qc_comp.append(ctrl_U_elem, range(qubit_size_tot))
                
    return qc_comp

def multiplexed_B(ctrl_size:int, select_size: int, coeffs: list):
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
        weights = np.zeros(2**ctrl_size, dtype = float)
        for j in range(cols):
            state_vector[j] = normalized_coeffs_i[j]
            weights[j] = normalized_coeffs_i[j]**2
        
        state_vector = Statevector(state_vector)
        qc_temp = QuantumCircuit(ctrl_size)
        qc_temp.append(StatePreparation(state_vector), range(ctrl_size))
        # print(qc_temp)
        control_values = bin(i)[2:].zfill(select_size)
        ctrl_qc_temp = qc_temp.to_gate().control(num_ctrl_qubits=select_size, ctrl_state=control_values)
        sel_index = list(range(ctrl_size, ctrl_size + select_size))
        ctrl_index = list(range(ctrl_size))
        qc_B.append(ctrl_qc_temp, sel_index + ctrl_index)
        
    # qc_B = transpile(qc_B, basis_gates=['x', 'y', 'z', 'rx', 'ry', 'h', 'cx'], optimization_level=1)
    return qc_B



def channel_to_LCU (ensem: channel_ensemble, structure = 'basic', opt = 'No') -> list:
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

    sys_size = max([ms.size for ms in channel])
    ## 2* select size is the count of Kraus operators (j)
    select_size = int(np.ceil(np.log2(len(channel))))
    ### 2**ctrl size is the count of Pauli terms (k) in each Kraus operator.  k indexes before j.
    ctrl_size = max([ms.ctrl_size for ms in channel])

    ## Channels are viewed as Kraus operators (matrixsums)
    coeff_sums = [ms.pauli_norm() for ms in channel]
    
    sys = QuantumRegister(sys_size, 'sys')
    
    ## Prepare the superposition state 
    LCU_ini = prep_sup_state(coeff_sums)
    
    ## First construct elementary block encodings then select them using select register. 
    if structure == 'basic':
        ctrl = QuantumRegister(ctrl_size, 'c')
        select = QuantumRegister(select_size, 's')
        qc = QuantumCircuit(select, ctrl, sys)
        qc.compose(LCU_ini, qubits=list(range(select_size)), inplace=True) 
        for i, ms in enumerate(channel):
            ctrl_value = bin(i)[2:].zfill(select_size)
            ctrl_v_rev = ctrl_value[::-1]
            qc_be_ms = BlockEncoding(ms).circuit(opt = opt)
            print(qc_be_ms.draw())
            ctrl_size_ms = qc_be_ms.num_qubits - sys_size
            # U_be_ms = qc_be_ms.to_gate().control(num_ctrl_qubits=select_size, ctrl_state=ctrl_value)
            U_be_ms = qc_be_ms.to_gate().control(num_ctrl_qubits=select_size, ctrl_state=ctrl_v_rev)
            qc.append(U_be_ms, qargs = list(select) + list(ctrl[:ctrl_size_ms]) + list(sys))
            qubit_indexes = [list(range(ctrl_size)), list(range(ctrl_size, ctrl_size + select_size)), 
                             list(range(ctrl_size + select_size, ctrl_size + select_size + sys_size))]
    elif structure == 'opt':
        circuits = []
        qc = QuantumCircuit(2 * select_size + ctrl_size + sys_size)
        
        select_indexes = [2 * j for j in range(select_size)]
        anc_indexes = [2 * j + 1 for j in range(select_size)]
        ctrl_indexes = list(range(2 * select_size, 2 * select_size + ctrl_size))
        sys_indexes = list(range(2 * select_size + ctrl_size, 2 * select_size + ctrl_size + sys_size))
        qc.compose(LCU_ini, qubits = select_indexes, inplace = True)
        for i, ms in enumerate(channel):
            qc_be_ms = BlockEncoding(ms).circuit(opt = opt)
            circuits.append(qc_be_ms)
        qc, tcount, mccount, cxcount = mulplex_U_opt(qc, circuits, select_size, ctrl_size , sys_size)
        qubit_indexes = [select_indexes, anc_indexes, ctrl_indexes, sys_indexes]
    # qc.save_statevector('final_state') #type: ignore
    # qubit_regs = [ctrl, select, sys]
    return [qc, qubit_indexes]


### Preparation for AA: channel_to_LCU operates over encoded states 
### Approximation: there are r groups of ancillae, at most h of them are non-zero.
def channel_to_LCU_encoded(ensem: channel_ensemble, r: int, h: int):
    channel = ensem.channels[0][1]
    sys_size = int(np.log2(channel[0].instances[0][0].to_operator().dim[0]))
    logr = int(np.ceil(np.log2(r + 1)))
    zerow_size = logr * h 
    ### There are at most h non-zero groups. 
    select_size = int(np.ceil(np.log2(len(channel))))
    ctrl_size = max([int(np.ceil(np.log2(len(ms.instances)))) for ms in channel])
    ## Store elementary matrices and their coefficients. 
    coeff_channel = [
        [0.0 for _ in range(len(channel[0].instances))]
        for _ in range(len(channel))
    ]
    mats_channel = [
        []
        for _ in range(len(channel))
    ]
    coeff_sums = []
    a00_amp = 0.0
    s0_amp = 0.0
    for i, ms in enumerate(channel):
        coeff_ms_sum = 0.0
        assert isinstance(ms, Matrixsum)
        for j, inst in enumerate(ms.instances):
            mat, coeff = inst 
            assert isinstance(coeff, float)
            if mat.expr == 'I' * mat.size:
                a00_amp = coeff

            coeff_channel[i][j] = coeff
            coeff_ms_sum += coeff
            if isinstance(mat, PauliAtom):
                mats_channel[i].append(SparsePauliOp([mat.expr], np.array([mat.phase])))
            else:
                mats_channel[i].append(mat.to_operator().data)
        coeff_sums.append(coeff_ms_sum)
        
        if i == 0:
            s0_amp = coeff_ms_sum

    total_coeff_sum = sum(np.array(coeff_sums)**2)
    
    zero_prob = (a00_amp * s0_amp / total_coeff_sum)  # type: ignore
    coeff_sums_0 = np.sqrt(s0_amp * (s0_amp - a00_amp))
    coeff_sums[0] = coeff_sums_0

    coeff_channel[0][0] = 0.0
    
    for j, inst in enumerate(channel[0].instances):
        coeff_channel[0][j] = coeff_channel[0][j] * s0_amp / coeff_sums_0

    ## Prepare a superposition state without |00> term 

    new_qc_ini = prep_sup_state(coeff_sums)
    
    new_multiplexed_B = multiplexed_B(ctrl_size, select_size, coeff_channel)
    
    qc_sub = QuantumCircuit(select_size + ctrl_size)
    qc_sub.compose(new_qc_ini, qubits = list(range(ctrl_size, select_size + ctrl_size)), inplace=True)
    qc_sub.compose(new_multiplexed_B, inplace=True)
    
    new_multiplexed_U = multiplexed_U(ctrl_size, select_size, sys_size, mats = mats_channel)

    greg = QuantumRegister(zerow_size, 'g')
    select = QuantumRegister(select_size * h, 's')
    ctrl =  QuantumRegister(ctrl_size * h, 'c')
    anc = QuantumRegister(1, 'a')
    sys = QuantumRegister(sys_size, 'sys')
    total_regs = [greg, anc, ctrl, select, sys]
    g_state, qc_zerow = create_zerow_comp_state(r, h ,zero_prob)
    
    return qc_zerow, new_multiplexed_B, qc_sub, new_multiplexed_U, total_regs

def get_indices(i: int, r: int, h: int, total_regs: list):
    greg, anc, ctrl, select, sys = total_regs
    ctrl_size = len(ctrl) // h
    select_size = len(select) // h
    logr = int(np.ceil(np.log2(r + 1)))
    g_idx = list(greg[i * logr: (i + 1) * logr])
    c_idx = list(ctrl[i * ctrl_size : (i + 1) * ctrl_size])
    s_idx = list(select[i * select_size : (i + 1) * select_size])
    return g_idx, c_idx, s_idx

def apply_encoding(qc, qc_sub, total_regs, r, h, inverse = False):
    greg, anc, ctrl, select, sys = total_regs
    
    if inverse == True:
        qc_tobe_encoded = controls_on_r(qc_sub.inverse(), r)
        print(qc_sub.inverse().draw())
    else:
        qc_tobe_encoded = controls_on_r(qc_sub, r)
        print(qc_sub.draw())
    for i in range(h):
        g_idx, c_idx, s_idx = get_indices(i,r, h,  total_regs)
        
        if qc_tobe_encoded.num_qubits == len(g_idx) + 1 + len(c_idx) + len(s_idx):
            qc.compose(qc_tobe_encoded, qubits = list(g_idx) + [anc[0]] + list(c_idx) + list(s_idx), inplace = True)
        else:
            qc.compose(qc_tobe_encoded, qubits = list(g_idx) + [anc[0]] + list(c_idx) + list(s_idx) + list(sys), inplace = True)

def apply_reflection(qc, qubits):
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)

## Directly encode Amplitude Amplification into the LCU circuit
def channel_to_LCU_AA(ensem: channel_ensemble, r: int, h: int): 

    qc_zerow, mul_B , qc_sub, mul_U, total_regs = channel_to_LCU_encoded(ensem, r, h)
    
    ## A superlong circuit for AA
    ## F = - W (I-2P1) W^(-1) (I-2P0) W 
    ## Equivalent: F = - MB^(-1)MU E(proj_|000>)E^(-1) MU^(-1) E (proj__|00>)E^(-1) MU E. 
    greg, anc, ctrl, select, sys = total_regs
    qc = QuantumCircuit(greg, anc, ctrl, select, sys)

    qc.compose(qc_zerow, qubits = list(greg), inplace = True)
    apply_encoding(qc, qc_sub, total_regs, r, h,  inverse = False)
    apply_encoding(qc, mul_U, total_regs, r, h, inverse = False)
    # apply_encoding(qc, qc_sub, total_regs, r, h, inverse = True)
    # proj_qubits = list(greg) + list(ctrl)
    # apply_reflection(qc, proj_qubits)

    # apply_encoding(qc, qc_sub, total_regs, r, h, inverse = False)
    # apply_encoding(qc, mul_U, total_regs, r, h, inverse = True)
    # apply_encoding(qc, qc_sub, total_regs, r, h, inverse = True)
    # proj_qubits = list(greg) + list(ctrl) + list(select)
    # apply_reflection(qc, proj_qubits)

    # apply_encoding(qc, qc_sub, total_regs, r, h, inverse = False)
    # apply_encoding(qc, mul_U, total_regs, r, h, inverse = False)
    apply_encoding(qc, mul_B, total_regs, r, h, inverse = True)
    # qc.global_phase += np.pi
    qc.save_statevector('final_state') #type: ignore
    return qc, total_regs

### Approximating states
def approx_state(sv, tol: float = 1e-8):
    """
    Approximate a vector/matrix by removing small amplitudes
    """
    sv_array = sv.data.copy()
    sv_array.real[np.abs(sv_array.real) < tol] = 0.
    sv_array.imag[np.abs(sv_array.imag) < tol] = 0. 
    return Statevector(sv_array)

def get_postmeas_density(final_sv: Statevector, qubit_regs: list):
    """
    Get the post-measurement density matrix after measuring the control qubits to |0>
    """
    if len(qubit_regs) == 3:
        ctrl, select, sys = qubit_regs
        ctrl_size, select_size, sys_size = len(ctrl), len(select), len(sys)
    elif len(qubit_regs) == 4:
        select, anc, ctrl, sys = qubit_regs
        select_size, anc_size, ctrl_size, sys_size = len(select), len(anc), len(ctrl), len(sys)
    else:
        raise ValueError("Unexpected register layout, qubit_regs should be either [ctrl, select, sys] or [select, anc, ctrl, sys]")

    if len(qubit_regs) == 3:
        # final_sv is prepared as: sys_state.tensor(ctrl_state.tensor(sel_state))
        # Flattened index: idx = ((sys_idx << ctrl_size) + ctrl_idx) << select_size + sel_idx
        data = np.asarray(final_sv.data, dtype=complex)
        dim_sys = 1 << sys_size
        dim_sel = 1 << select_size
        stride_sys = 1 << (ctrl_size + select_size)

        # amplitudes of |sys>\otimes|sel> conditioned on ctrl=0...0
        post_amp_sys_sel = np.zeros((dim_sys, dim_sel), dtype=complex)
        
        for sys_idx in range(dim_sys):
            base = sys_idx * stride_sys
            for sel_idx in range(dim_sel):
                post_amp_sys_sel[sys_idx, sel_idx] = data[base + sel_idx]

        # Explicitly partial-trace over select register:
        # rho_sys[a,b] = sum_s amp[a,s] * conj(amp[b,s])
        system_density_np = np.einsum('as,bs->ab', post_amp_sys_sel, post_amp_sys_sel.conj())
        trace_val = np.trace(system_density_np)
        
        if abs(trace_val) > 1e-14:
            system_density_np = system_density_np / trace_val
        system_density = DensityMatrix(system_density_np)
        
    elif len(qubit_regs) == 4:
        # Optimized layout: qubit_regs = [select_index, anc_index, ctrl_index, sys_index]
        # Extract amplitudes from entries where anc_index and ctrl_index are fixed to |0...0>,
        # keeping only select+sys degrees of freedom.
        data = np.asarray(final_sv.data, dtype=complex)
        dim_sys = 1 << sys_size
        dim_sel = 1 << select_size

        post_amp_sys_sel = np.zeros((dim_sys, dim_sel), dtype=complex)

        def place_bits(base_idx: int, local_idx: int, reg_qubits: list[int]) -> int:
            out = base_idx
            for bit_pos, qubit_id in enumerate(reg_qubits):
                if ((local_idx >> bit_pos) & 1) == 1:
                    out |= (1 << qubit_id)
            return out

        # anc_index and ctrl_index bits are fixed to 0 by construction (base_idx = 0)
        for sys_idx in range(dim_sys):
            for sel_idx in range(dim_sel):
                flat_idx = 0
                flat_idx = place_bits(flat_idx, sel_idx, select)
                flat_idx = place_bits(flat_idx, sys_idx, sys)
                post_amp_sys_sel[sys_idx, sel_idx] = data[flat_idx]

        # Partial trace over select register
        system_density_np = np.einsum('as,bs->ab', post_amp_sys_sel, post_amp_sys_sel.conj())
        trace_val = np.trace(system_density_np)
        if abs(trace_val) > 1e-14:
            system_density_np = system_density_np / trace_val
        system_density = DensityMatrix(system_density_np)

    
    return system_density
def construct_qobj_lind(Lind: Lindbladian, dim_sys: int):
    """
    Construct the Qobj representation of the Lindbladian superoperator.
    """
    H = Lind.H.eff_op()
    H_qobj = Qobj(H, dims = [[2] * dim_sys, [2] * dim_sys])
    L_qobj_list = []
    for L in Lind.L_list:
        L_qobj_list.append(Qobj(L.eff_op(), dims = [[2] * dim_sys, [2] * dim_sys]))
    
    return H_qobj, L_qobj_list


def simulate_circuit(circuit: QuantumCircuit, ini_state: Statevector, qubit_regs: list):
    simulator = AerSimulator()
    qc_test = deepcopy(circuit)
    
    ## Distinguish the type of registers using length of qubit_regs
    qc_test = transpile(qc_test, simulator, optimization_level=1)
    qc_test.save_statevector(label = 'final_state')
    ## Extract system density for comparison
    result_job = simulator.run(qc_test, shots = 1, initial_state=ini_state).result().data(0)
    print('Simulation done, extracting final state...   ')
    final_sv = result_job['final_state']
    final_sv = approx_state(Statevector(final_sv), tol=1e-6)
    system_density_final = get_postmeas_density(final_sv, qubit_regs).data
    
    return DensityMatrix(system_density_final)


def simulate_circuit_opt(circuit: QuantumCircuit, ini_state: Statevector, qubit_regs: list):
    simulator = AerSimulator()
    qc_test = deepcopy(circuit)
    
    ## Distinguish the type of registers using length of qubit_regs
   
    
    return DensityMatrix(system_density_final)




if __name__ == "__main__":
   
    test_case = 4

    ### Test II: A simple Lindbladian 
    ### A transverse field Ising model Lindbladian
    if test_case == 2:
        H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
        gamma = np.sqrt(0.1)/2 
        L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
        delta_t = 0.05
        TFIM_lind = Lindbladian(H, L_list)
        
        channel_Lind, success_prob_th, coeff_sum = Lindblad_to_channel(TFIM_lind, delta_t)
        # print(success_prob_th)
        channel_Lind = channel_Lind.channels[0][1]
        ms = channel_Lind[0]
        H_qobj, L_qobj_list = construct_qobj_lind(TFIM_lind, dim_sys = 3)  
        
        LCU, qubit_regs = channel_to_LCU(channel_ensemble([channel_Lind]))
        qubit_size = [len(qreg) for qreg in qubit_regs]
        norm = channel_norm_zero(channel_Lind, Statevector.from_label('0' * qubit_size[2]))
        ini_state = Statevector.from_label('0' * sum(qubit_size))
        system_density_final = simulate_circuit(LCU, ini_state = ini_state, qubit_regs = qubit_regs) 
    
        ini_state_qutip = tensor(basis(2,0), basis(2,0), basis(2,0))
        baseline_density = simulate_lindblad(H_qobj, L_qobj_list, ini_state_qutip, delta_t, 8)

        diff = system_density_final - DensityMatrix(baseline_density)

        print("Norm difference:", np.linalg.norm(diff, ord = 'nuc')/2)

        pnorm = Lindblad_paulinorm(TFIM_lind)
        print(f"Lindbladian Pauli norm: {pnorm}")
        # print(success_prob)
    

    ### Test III: Encoded LCU circuit for AA
    if test_case == 3:
        H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
        gamma = np.sqrt(0.1)/2 
        L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
        delta_t = 0.1
        TFIM_lind = Lindbladian(H, L_list)

        r = 1
        h = 1
        success_prob_sym = probs_from_lindblad(TFIM_lind)
        equation = sp.Eq(success_prob_sym, 4**(-1/r))
        initial_guess = 0.05
        delta_t = sp.nsolve(equation, sp.symbols('delta_t', real = True, positive = True), initial_guess)
        channel_Lind, success_prob_th, coeff_sum = Lindblad_to_channel(TFIM_lind, float(delta_t))
        channel_Lind = channel_Lind.channels[0][1]

        ## A single LCU circuit as baseline 
        LCU, qubit_regs = channel_to_LCU(channel_ensemble([channel_Lind]))
        print(LCU.draw())
        qubit_size = [len(qreg) for qreg in qubit_regs]
        norm = channel_norm_zero(channel_Lind, Statevector.from_label('0' * qubit_size[2]))
        ini_state = Statevector.from_label('0' * sum([len(qreg) for qreg in qubit_regs]))
        final_sv = simulate_circuit(LCU, ini_state = ini_state, qubit_regs = qubit_regs)
        
        # print("Norm of the channel:", norm)
        # print("Single Success probability:", success_prob)
        # print("Theoretical success prob for r iterations:", np.pow(success_prob, r))
        # print("Simulated evolution time:", delta_t * r)

        ## Repeat r times 

        LCU_encoded, qubit_regs = channel_to_LCU_AA(channel_ensemble([channel_Lind]), r, h)
        qubit_size = [len(qreg) for qreg in qubit_regs]
        norm = channel_norm_zero(channel_Lind, Statevector.from_label('0' * qubit_size[-1]))
        ini_state = Statevector.from_label('0' * sum(qubit_size))
        final_sv = simulate_circuit(LCU_encoded, ini_state = ini_state, qubit_regs = qubit_regs )



    
   


  