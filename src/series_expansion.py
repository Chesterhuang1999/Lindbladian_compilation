import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, SparsePauliOp, partial_trace
from qiskit.circuit.library import UnitaryGate, RYGate, XGate
import qiskit.qasm2
from qiskit_aer import  AerSimulator
from qiskit_aer.library import save_statevector, save_density_matrix
from qiskit.circuit.annotated_operation import AnnotatedOperation, ControlModifier
from qutip import Qobj, tensor, basis
from channel_IR import *
from subroutine import *
from baseline import simulate_lindblad 
from copy import deepcopy

from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm, cosm, sinm
import itertools

## Prepare the quadrature points and weights from a series of Legendre polynomials
def leg_vals(q, t):
    ### Prepare interpolation points and coeffs for the series expansion

    x_ori, w_ori = leggauss(q)

    ### Zeropoint and gaussian weights of scaled legendre polynomials
    x = np.array([t/2 * (xi + 1) for xi in x_ori])
    w = np.array([wi * t/2 for wi in w_ori])
    return x, w
def quadrature_points_N_weights(k: int, q: int, t: float, samp_pts:list):
    """
    Prepare the quadrature points and weights 
    according to the interval indicators samp_pts, i.e. the values of j1.... jk
    """
    x, w = leg_vals(q, t)
    assert len(samp_pts) == k
    ### Now, create the canonical quadratue points and weights:
    ### val_(j_k) = x_(j_k), weight_(j_k) = w_(j_k) j_k = 1....q
    ### val_(j_k, ...,j_(k-l)) = x_(j_(k-l)) * ... * x_(j_k) / t^(l), weight_(j_k, ...,j_(k-l)) = w_(j_(k-l)) * ... * w_(j_k) / t^(l)
    
    ### There are k points 
    new_x = []
    new_w = []
    for i, ind in enumerate(samp_pts):
        if i == 0:
            val = x[ind]
            weight = w[ind]
        else: 
            weight = val * w[ind] / t 
            val = val * x[ind] / t
            
        new_x.append(val)
        new_w.append(weight)
    return new_x, new_w

# def pcphase(phi:float, ctrl:QuantumRegister, sys:QuantumRegister):
#     qc= QuantumCircuit(ctrl, sys)
#     if len(ctrl) > 1:
#         qc.x(ctrl[1:])
#         qc.mcx(ctrl[1:], ctrl[0])
#         qc.x(ctrl[1:])
#         qc.rz(-2 * phi, ctrl[0])
#     else:
#         qc.rz(2 * phi, ctrl[0])
#     if len(ctrl) > 1:
#         qc.x(ctrl[1:])
#         qc.mcx(ctrl[1:], ctrl[0])
#         qc.x(ctrl[1:])
#     print(qc.draw())
#     return Operator(qc.to_gate()).data
        
def lcu_prepare_tree(weights):
    """
    Prepare the superposition state of LCU coefficients.
    weights: list of nonnegative numbers, sum(weights) = 1
    """
    n = int(np.log2(len(weights)))
    assert 2**n == len(weights)

    # qc = QuantumCircuit(n)

    def recurse(level, probs):
        if level == n:  
            return 
        
        half = len(probs) // 2

        p0 = sum(probs[:half])
        p1 = sum(probs[half:])
        if p0 + p1 == 0: 
            return ## Zero-prob branch, nothing to do 
        theta = 2 * np.arccos(np.sqrt( p0 /(p0 + p1)))
        t = n - level - 1
        qc = QuantumCircuit(t + 1)
        qc.ry(theta, t)
        
        # qc.x(t)
        qc_sub = recurse(level + 1, probs[:half])
        if qc_sub is not None:
            qc.x(t)
            u_sub = qc_sub.to_gate()
            u_sub = u_sub.control(1, ctrl_state = '1')
            qc.append(u_sub, [qc.qubits[-1]] + qc.qubits[:-1])
            qc.x(t)
        
        qc_sub = recurse(level + 1, probs[half:])
        if qc_sub is not None:
            u_sub = qc_sub.to_gate()
            u_sub = u_sub.control(1, ctrl_state = '1')
            qc.append(u_sub, [qc.qubits[-1]] + qc.qubits[:-1])
        return qc
    
    qc = recurse(0, weights)

    return qc

def mulplex_B(coeff_list: list, ctrl_size: int) -> QuantumCircuit:
    """
    Create the multiplexed B operator for the LCU circuit.
    coeff_list is the list of coefficients for each term in the LCU.
    """
    sum_coeff = sum([abs(c) for c in coeff_list])
    norm_coeffs = [abs(c)/sum_coeff for c in coeff_list]
    probs = np.zeros(2**ctrl_size, dtype = float)
    for i, nc in enumerate(norm_coeffs):
        probs[i] = nc
    
    qc = lcu_prepare_tree(probs) 
    return qc #type: ignore

def mulplex_U(mat_list: list, ctrl_size: int, sys_size: int) -> QuantumCircuit:
    """
    Create the multiplexed U operator for the LCU circuit.
    mat_list is the list of matrices for each term in the LCU.
    """
    qc = QuantumCircuit(ctrl_size + sys_size)
    
    for i, ms in enumerate(mat_list):
        if isinstance(ms, SparsePauliOp):
            pauli_op, phase = ms.paulis[0], ms.coeffs[0]

            qc_pauli = QuantumCircuit(pauli_op.num_qubits)
            qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
            qc_pauli.global_phase = np.angle(phase)
            qc_pauli = qc_pauli.decompose()
            U_elem = qc_pauli.to_gate()
            print(qc_pauli.draw())
        else:
            if ms.shape[0] < 2**sys_size:
                pad_size = 2**sys_size // ms.shape[0]
                ms = np.kron(ms, np.eye(pad_size))
        
        control_values =  bin(i)[2:].zfill(ctrl_size) 
        if len(qc_pauli.data) > 0: ### Identity is ignored 

            ctrl_U_elem = U_elem.control(num_ctrl_qubits = ctrl_size, ctrl_state = control_values)
            qc.append(ctrl_U_elem, range(ctrl_size + sys_size))
            # qc.append(ctrl_U_elem, list(range(sys_size, ctrl_size + sys_size)) + list(range(sys_size)))
            
    return qc

def apply_reflection(qc, qubits):
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
def block_encoding_matrixsum(L: Matrixsum):
    """
    Create the block-encoding circuit for the matrixsum term L.
    L can be the incoherent operator L_j, or the basic block-encoding gadget H in 
    the QSVT implementation of Hamiltonian simulation e^{Jt}. 
    """
    mat_list = []
    coeff_list = []
    if len(L.instances) == 1:
        matrix, coeff = L.instances[0]
        qc_pauli = QuantumCircuit(matrix.size)
        qc_pauli.append(Pauli(matrix.expr), range(matrix.size))
        qc_pauli.global_phase += matrix.phase
        return qc_pauli
    for matrix, coeff in L.instances:
        if isinstance(matrix, PauliAtom):
            mat_list.append(SparsePauliOp([matrix.expr], np.array([matrix.phase])))
        else:
            mat_list.append(matrix.to_operator().data)
        coeff_list.append(coeff)
    ctrl_size = int(np.ceil(np.log2(len(coeff_list))))
    sys_size = int(np.log2(mat_list[0].to_operator().dim[0]))
    
    ctrl = QuantumRegister(ctrl_size, 'ctrl')
    sys = QuantumRegister(sys_size, 'sys')
    # qc = QuantumCircuit(sys, ctrl)
    qc = QuantumCircuit(ctrl, sys)
    
    qc.compose(mulplex_B(coeff_list, ctrl_size), qubits = ctrl, inplace = True)
    qc.compose(mulplex_U(mat_list, ctrl_size, sys_size), qubits = ctrl[:] + sys[:], inplace = True)
    
    qc.compose(mulplex_B(coeff_list, ctrl_size).inverse(), qubits = ctrl, inplace = True)

    
    return qc

def apply_poly_phases(phi_values, gadget, qc: QuantumCircuit, anc, ctrl):
    """
    Apply projection-controlled phase rotations to make the QSVT circuit.
    """
    qc.h(anc[0])

    for i, phi in enumerate(reversed(phi_values)):
        if ctrl is not None:
            qc.x(ctrl)
            qc.mcx(ctrl, anc[0])
            # qc.x(ctrl)
        
        qc.rz(2 * phi, anc[0])
        if ctrl is not None: 
            # qc.x(ctrl)
            qc.mcx(ctrl, anc[0])
            qc.x(ctrl)

        if i % 2 == 0 and i != len(phi_values) - 1:
            qc.append(gadget, qc.qubits[1:])
         
        elif i % 2 == 1 and i != len(phi_values) - 1:
            qc.append(gadget.inverse(), qc.qubits[1:])
        
    qc.h(anc[0])

    return qc
def qsvt_Hamiltonian(J: Matrixsum, t: float, q: int, l: int):
    """
    Create the QSVT circuit for the Hamiltonian terms e^-iJt
    J is the coherent term: J = H - 1/2i sum L^dag L
    q is number of quadrature points,
    and l is truncation order for term J.
    """
    ### Create the block-encoding of J
    qc_basic = block_encoding_matrixsum(J)
    # QSVT_basic_instr = qc_basic.to_instruction(label = "QSVT_basic_gadget") 
    
    subnorm_fac = 0
    coeff_list = []
    for mat, coeff in J.instances:
        subnorm_fac += abs(coeff)
        coeff_list.append(abs(coeff))
    ctrl_size = int(np.ceil(np.log2(J.length)))
    sys_size = J.size
    anc = QuantumRegister(1, 'a')
    ctrl = QuantumRegister(ctrl_size, 'c')
    sys = QuantumRegister(sys_size, 's')
    ## An additional ancilla qubit for Hamiltonian evolution e^(-iHt), initialized in |+>
    
    N = 1000
    deg = 12
   
    cos_func = lambda x: np.cos(subnorm_fac * t * x)
    cos_phi_values, max_value_cos = phase_generator(cos_func, N ,deg)
    
    sin_func = lambda x: np.sin(subnorm_fac * t * x)
  
    sin_phi_values, max_value_sin = phase_generator(sin_func, N , deg - 1)
    
    cos_phi_values[0] += np.pi / 4 #type:ignore
    for i in range(1, len(cos_phi_values) - 1):
        cos_phi_values[i] += np.pi / 2 #type: ignore
    cos_phi_values[-1] += np.pi / 4 #type: ignore

    sin_phi_values[0] +=  np.pi / 4 #type: ignore
    for i in range(1, len(sin_phi_values) - 1):
        sin_phi_values[i] += np.pi / 2 #type: ignore
    sin_phi_values[-1] += np.pi / 4 #type: ignore

    QSVT_basic_gadget = qc_basic.to_gate(label = 'QSVT_basic_gadget')
    
    qc_sin = QuantumCircuit(anc, ctrl, sys)
    qc_cos = QuantumCircuit(anc, ctrl, sys)

    qc_sin = apply_poly_phases(sin_phi_values, QSVT_basic_gadget, qc_sin, anc, ctrl)
    qc_sin.global_phase += - np.pi / 2
    qc_cos = apply_poly_phases(cos_phi_values, QSVT_basic_gadget, qc_cos, anc, ctrl)
    
    
    
    U_ctrl_cos = qc_cos.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '0')
    U_ctrl_sin = qc_sin.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1')

    # op_cos = AnnotatedOperation(qc_cos.to_instruction(),
    #                             ControlModifier(1, '0'))
    
    # op_sin = AnnotatedOperation(qc_sin.to_instruction(),
    #                             ControlModifier(1, '1'))
                                

    ## Prepare a LCU circuit for e^(iHt) = cos(Ht) - isin(Ht)
    sel = QuantumRegister(1, 'sel')
    qc_main = QuantumCircuit(sel, anc, ctrl, sys) 

    qc_main.ry(np.pi/2, sel) # Prepare coeff (1/sqrt(2), 1/sqrt(2))
    # qc_main.append(op_cos, qc_main.qubits)
    # qc_main.append(op_sin, qc_main.qubits)

    qc_main.append(U_ctrl_cos, qc_main.qubits)
    qc_main.append(U_ctrl_sin, qc_main.qubits)

    qc_main.ry(-np.pi/2, sel)


    return qc_main, max_value_cos + max_value_sin


#### Construct the Lindbladian evolution circuit via higher-order series expansion
def create_single_kraus(
        k: int, index: list, L_list: list, 
        t: float, ctrl_size: int, sys_size:int,
        K1: int, eff_H: Matrixsum
        ):
    """ 
    Create the LCU encoding circuit by 
    Concatenating the block-encoding circuits for 
    each coherent gadget e^{J delta_t} and decay term L_j. 
    A_j = sqrt(w_(jk) * ... * w_(jk,...j1)) e^J(t - x_(jk))L_(lk)..... L_(l1)

    Parameters: 
    k (int): expansion order
    index (list): indexes of quadrature points and decay terms
    L_list (list): list of decay terms
    eff_H: effective Hamiltonian J
    ctrl_size, sys_size: control and system register sizes
    K1: truncation order for J's Taylor series
    t: evolution time
    Returns:
    qc_elem: the cascaded block-encoding circuit for Kraus operator A_j
    coeff_sum_index sum of coefficients in Kraus A_j

    """
    ## 'indexes of L_j and quadrature points j_k...j_1'
    l_list = index[1: k + 1]
    j_list = index[k + 1: 2 * k + 1]
    coeff_sum_index = 1.0 ## Set coeff sum value
    ## Generate quadrature points and weights based on 
    points, weights = quadrature_points_N_weights(k, q, t, j_list)
    points.append(0) ## Add minimum point 0
    weights_prod = np.prod(weights)
    ## Mul coefficient 'sqrt(w_(jk) * ... * w_(jk,...j1))'
    coeff_sum_index *= np.sqrt(weights_prod)
    
    ## Sum up the control register sizes
    ctrl_reg_size = []
    for j in range(k + 1):
        ctrl_reg_size.append(ctrl_size)
        L_j = L_list[l_list[k - j - 1]]
        if j < k:
            ctrl_size_lj = int(np.ceil(np.log2(L_j.length)))
            ctrl_reg_size.append(ctrl_size_lj)
    total_ctrl_size = sum(ctrl_reg_size)
    total_ctrl_reg = QuantumRegister(total_ctrl_size, 'ctrl')
    system_reg = QuantumRegister(sys_size, 'sys')
    qc_elem = QuantumCircuit(total_ctrl_reg, system_reg)
    ctrl_offset = 0
    ## Concatenate the block-encoding circuits from back to forth
    for j in range(k + 1):
        ## calculate each time interval
        if j == k:
            delta_t = t - points[0]
        else:
            delta_t = points[k - j - 1] - points[k - j] ## The
        
        ### Concatenate the block-encoding for coherent term e^{J delta_t}
        qc_C, max_value_C = qsvt_Hamiltonian(eff_H, delta_t, q, K1)
        qc_elem.compose(qc_C, qubits = qc_elem.qubits[ctrl_offset: ctrl_offset + ctrl_reg_size[2 * j]] 
                        + qc_elem.qubits[total_ctrl_size:], inplace = True)
        ctrl_offset += ctrl_reg_size[2 * j]
        coeff_sum_index *= max_value_C
        ### Concatenate the block-encoding for incoherent term L_(l_(k-j))
        if j < k:
            L_j = L_list[l_list[k - j - 1]]
            coeff_sum_index *= L_j.pauli_norm() ## sum of coefficients 
            qc_L = block_encoding_matrixsum(L_j)
            
            qc_elem.compose(qc_L, qubits = qc_elem.qubits[ctrl_offset: ctrl_offset + ctrl_reg_size[2 * j + 1]] 
                        + qc_elem.qubits[total_ctrl_size:], inplace = True)
            ctrl_offset += ctrl_reg_size[2 * j + 1]
    return qc_elem, coeff_sum_index, total_ctrl_size
def create_index_set(k: int, q: int, m: int): 
    """
    Create the index set for the k-th order term in the series expansion.
    The tuple is [k, l1...lk, j1...jk], l1... lk in [0, m-1], j1...jk in [0, q-1]
    """
    index_set = []
    iter_l = itertools.product(range(m), repeat = k)
    
    for l_is in iter_l: 
        iter_j = itertools.product(range(q), repeat = k)
        for j_is in iter_j: 
            index_set.append([k] + list(l_is) + list(j_is))

    return index_set
def construct_terms_coherent(J: Matrixsum, K1: int, t: float):
    sum_of_terms = 0
    sum_of_J = J.identity(J.size)
    for order in range(1, K1 + 1):
        if order == 1: 
            sum_of_terms += J.length
            product_J = deepcopy(J)
            product_J.mul_coeffs(t)
            sum_of_J = sum_of_J.add(deepcopy(J))
        else: 
            product_J = matsum_mul(product_J, J)
            product_J.mul_coeffs(t / order)
            sum_of_J.add(product_J)
            sum_of_terms += product_J.length
    return sum_of_terms, sum_of_J
def higher_order_Lind_expansion(Lind: Lindbladian, K: int, q: int, t: float, K1: int = 1):
    """
    Create the higher-order Lindblad evolution expansion circuit
    using Kth order series expansion with q quadrature points.
    """
    ### Zeroth order term: e^Jt
    effective_H = Lind.effective_H()
    m = len(Lind.L_list)
    sys_size = effective_H.size
    sum_of_kraus = 0

    # length_coherent, approx_eff_H = construct_terms_coherent(effective_H, K1, t)
    # ctrl_size_coh = int(np.ceil(np.log2(length_coherent)))
    
    qc_ensemble = []
    coeff_sum_total = []
    ctrl_sizes = []
    ctrl_size_max = 0
    kraus_count = 1

    qc_info = qsvt_Hamiltonian(effective_H, t, q, K1)
    qc_coh_ctrl_size = qc_info[0].num_qubits - sys_size
    
    coeff_sum_total.append(qc_info[1])
    ctrl_size_max = max(ctrl_size_max, qc_coh_ctrl_size)
    
    ## Prepare the component circuit for each Kraus operator 
    for k in range(1, K + 1):
        
        index_set_k = create_index_set(k, q, m)
        
        sum_of_kraus += len(index_set_k)

        ## Each Kraus operator is constructed from an index tuple
        for index in index_set_k:
            kraus_count += 1
            assert index[0] == k and len(index) == 2 * k + 1
            
            qc_elem, coeff_sum_index, elem_ctrl_size = create_single_kraus(k, index, Lind.L_list, t, 
                                                    qc_coh_ctrl_size, sys_size, K1, effective_H)
            ctrl_size_max = max(elem_ctrl_size, ctrl_size_max)
            qc_ensemble.append(qc_elem)
            coeff_sum_total.append(coeff_sum_index)
            ctrl_sizes.append(elem_ctrl_size)
   
    
    sel_size = int(np.ceil(np.log2(kraus_count))) 
    
    select_reg = QuantumRegister(sel_size, 'sel')
    control_reg = QuantumRegister(ctrl_size_max, 'ctrl')
    
    system_reg = QuantumRegister(sys_size, 'sys')

    qc_main = QuantumCircuit(select_reg, control_reg, system_reg)
    
    ## Prepare superposition over selection register
    coeff_sum_square = sum([abs(c)**2 for c in coeff_sum_total])
    norm_coeffs = [abs(c)**2 / coeff_sum_square for c in coeff_sum_total]
    weights = np.zeros(2**sel_size, dtype = float)
    for i, nc in enumerate(norm_coeffs):
        weights[i] = nc
    
    qc_prep = lcu_prepare_tree(weights)
    qc_main.compose(qc_prep, qubits = qc_main.qubits[:sel_size], inplace = True)
    
    
    ### Append A_0 = e^{Jt} circuit
    qc_coh_control = qc_info[0].to_gate().control(num_ctrl_qubits = sel_size, ctrl_state = '0' * sel_size)
    qc_main.append(qc_coh_control, qargs = qc_main.qubits[:sel_size] + qc_main.qubits[sel_size:sel_size + qc_coh_ctrl_size] 
                    + qc_main.qubits[sel_size + ctrl_size_max:])
    print("A_0 construct complete")
    ### Append each Kraus operator circuit
    for ind in range(1, kraus_count):
        
        qc_elem = qc_ensemble[ind -1]
        ctrl_size = ctrl_sizes[ind - 1]
        select_value = bin(ind)[2:].zfill(sel_size)
        qc_elem_control = qc_elem.to_gate().control(num_ctrl_qubits = sel_size, ctrl_state = select_value)

        qc_main.append(qc_elem_control, qargs = qc_main.qubits[:sel_size] + qc_main.qubits[sel_size: sel_size + ctrl_size]
                        + qc_main.qubits[sel_size + ctrl_size_max:])
        print(f"A_{ind} construct complete")
    reg_sizes = [sel_size, ctrl_size_max, sys_size]
    filename = f'circuits/TFIM_lind_K{K}_q{q}_t{t}.qasm'
    with open(filename, 'w') as f: 
        qiskit.qasm2.dump(qc_main, f)
    
    return qc_main, reg_sizes

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
def projection_op(op: Operator, dim: int, dim0: int):
    proj_0 = Operator.from_label('0' * dim0)
    iden = Operator.from_label('I' * (dim - dim0))

    proj_full = iden.tensor(proj_0)

    projected_op = np.dot(np.dot(proj_full, op), proj_full)
    projected_op_reduced = np.zeros((2**(dim - dim0), 2**(dim - dim0)), dtype = complex)
    for i in range(2**(dim - dim0)):
        for j in range(2**(dim - dim0)):
            projected_op_reduced[i, j] = projected_op[i * 2**dim0, j * 2**dim0]

    return Operator(zero_small_complex(projected_op_reduced, tol = 1e-6))

def projection_vec(sv: Statevector, dim: int, dim0: int):
    proj_0 = Operator.from_label('0' * dim0)
    iden = Operator.from_label('I' * (dim - dim0))

    proj_full = iden.tensor(proj_0) 
    projected_vec = np.dot(proj_full, sv)

    projected_vec = projected_vec[::2**dim0]

    return Statevector(zero_small_complex(projected_vec, tol = 1e-6))

def simulate_circuit_qsvt(qc: QuantumCircuit, ini_state):

    N = 10000
    simulator = AerSimulator()
    ancilla_regs = []
    for qbit in qc.qubits:
        c = qbit._register
        if c.name != 's':
            ancilla_regs.append(qbit)
    qc_sim = QuantumCircuit(qc.qubits, qc.clbits)

    qc_sim.initialize(ini_state)
    qc_sim.compose(qc, qc.qubits, qc.clbits, inplace = True)

    qc_sim = transpile(qc_sim, simulator)
    clbits = qc_sim.num_clbits
    
    ### Sim task I: Get final statevector
    
    result = simulator.run(qc_sim, shots = 1).result()

    final_state = result.data()['final_state']

    final_state_sys, norm = projection_vec(final_state, qc_sim.num_qubits, len(ancilla_regs))

    success_prob = norm
    
    ### Sim task II: Calculate success probability
    
    return final_state_sys, success_prob
def simulate_circuit_TN(qc: QuantumCircuit, ini_state: Statevector, reg_sizes: list):
    """
    Construct a tensor network simulation of the circuit qc with initial state ini_state.
    """
    sel_size, ctrl_size, sys_size = reg_sizes
    ancilla_size = sel_size + ctrl_size
    simulator = AerSimulator(method = 'matrix_product_state')
    simulator.set_options(matrix_product_state_max_bond_dimension=128)
    qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
    qc_sim.initialize(ini_state)
    qc_sim.compose(qc, qc.qubits, qc.clbits, inplace = True)
    
    qc_sim = transpile(qc_sim, simulator, optimization_level = 1)
    anc_output = ClassicalRegister(ctrl_size, 'c_out')
    qc_sim.add_register(anc_output)
    for i in range(ctrl_size):
        qc_sim.measure(qc_sim.qubits[i + sel_size], anc_output[i])
   
    with qc_sim.if_test((anc_output, 0)):
        qc_sim.save_density_matrix(qubits = list(range(sel_size)) + list(range(ancilla_size, ancilla_size + sys_size)), label = 'final_sys_dm') # type: ignore

    repeat = 10000
    count = 0
    for i in range(repeat):
        result = simulator.run(qc_sim, shots = 1).result()
        if list(result.get_counts().keys())[0] == '0'*ctrl_size:
            count += 1
            raw_rho = result.data()['final_sys_dm']
            final_rho = raw_rho 
    

    final_rho_sys = partial_trace(final_rho, list(range(sel_size)))
            
    return final_rho_sys
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

if __name__ == "__main__":
    K = 1
    q = 2
    t = 1
    test_case = 1
    H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    gamma = np.sqrt(0.1)/2 
    L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    delta_t = 0.1
    TFIM_lind = Lindbladian(H, L_list)
    if test_case == 1:

        H = [('YI', -0.5),('XY', -1)]
        L_list = []
        TFIM_lind = Lindbladian(H, L_list)
        eff_H = TFIM_lind.H.eff_op()
        H_evo = Operator(expm(-1j * eff_H.data * t)) #type: ignore
        ini_state = Statevector.from_label('+0')
        final_state_baseline = normalize(ini_state.evolve(H_evo))

        qc, max_value = qsvt_Hamiltonian(TFIM_lind.H, t, q, l = 2)
        H_evo_be = Operator(qc.to_gate())
        proj_H_evo_be = projection_op(H_evo_be, qc.num_qubits, qc.num_qubits -TFIM_lind.H.size)
        
        ini_state = Statevector.from_label('+0' + '0' * (qc.num_qubits - 2))
        final_state_sys = ini_state.evolve(qc)
        final_state_sys = normalize(projection_vec(final_state_sys, qc.num_qubits, qc.num_qubits - TFIM_lind.H.size))

        diff = DensityMatrix(final_state_sys) - DensityMatrix(final_state_baseline)
        err = np.linalg.norm(diff, ord = 'nuc') / 2
        print(err)

    elif test_case == 2:
        H_eff = TFIM_lind.effective_H()
        qc, max_value = qsvt_Hamiltonian(H_eff, delta_t, q,  2)
        H_evo = Operator(expm(-1j * H_eff.eff_op() * delta_t)) #type: ignore
        ini_state_qsvt = Statevector.from_label('+00' + '0' * (qc.num_qubits - 3)) 
        ini_state_baseline = Statevector.from_label('+00')
        final_state_baseline = normalize(ini_state_baseline.evolve(H_evo))
        print("Baseline:")
        print(final_state_baseline)
        final_state_sys = ini_state_qsvt.evolve(qc)
        final_state_sys = normalize(projection_vec(final_state_sys, qc.num_qubits, qc.num_qubits - 3))
        print("QSVT final state:")
        print(final_state_sys)
        diff = DensityMatrix(final_state_sys) - DensityMatrix(final_state_baseline)
        err = np.linalg.norm(diff, ord = 'nuc') / 2
        print(f"Simulate error: {err}")
    
    elif test_case == 3:
        qc, reg_sizes = higher_order_Lind_expansion(TFIM_lind, K, q, t, K1 = 2)
        print("Circuit model:",qc.draw())
        print("Size of circuit:", reg_sizes)
        ini_state = Statevector.from_label('+00' + '0' * (reg_sizes[0] + reg_sizes[1]))
        final_dens_sys = simulate_circuit_TN(qc, ini_state, reg_sizes)
        
        print("Simulated final density:", final_dens_sys)
        H_qobj, L_qobj_list = construct_qobj_lind(TFIM_lind, TFIM_lind.H.size)
        qplus = (basis(2,0) + basis(2,1)).unit()
        
        ini_state_qobj = tensor(qplus, basis(2, 0), basis(2,0))
    
        
        final_dens = simulate_lindblad(H_qobj, L_qobj_list, ini_state_qobj, delta_t, r = 10)
        
        purity = np.trace(final_dens@final_dens)
        
        print("Baseline final density:", final_dens)