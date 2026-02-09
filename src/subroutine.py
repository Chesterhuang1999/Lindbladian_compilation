import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from copy import deepcopy
from itertools import combinations
import sympy as sp
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, Pauli
from qiskit.circuit.library import StatePreparation, XGate
from qiskit_aer import noise, AerSimulator

from qsppack.utils import cvx_poly_coef, chebyshev_to_func
from qsppack.solver import solve
from channel_IR import * 
from pyqsp.angle_sequence import QuantumSignalProcessingPhases

from numpy.polynomial.legendre import leggauss

import itertools
def probs_from_lindblad(Lind: Lindbladian):
    
    delt = sp.symbols('delta_t', real = True, positive = True)
    channels = []
    iden = PauliAtom('I' * int(np.log2(Lind.H.instances[0][0].to_operator().dim[0])), phase = 1.0)
    kraus_0_basis = Matrixsum([(iden, 1.0)])

    H_copy = deepcopy(Lind.H)
    
    H_norm = H_copy.pauli_norm()
    total_norm = []
    kraus0_norm = H_norm * delt + 1
    Ls_copy = []

    for Ls in Lind.L_list:
        Ls_copy.append(deepcopy(Ls))
    for i in range(len(Ls_copy)):
        Ls_norm = 0.0
        Ls_product = matsum_mul(Ls_copy[i].adj(), Ls_copy[i])
        for mat, coef in Ls_product.instances:
            if mat.expr == 'I' * mat.size:
                Ls_norm += -0.5 * coef * delt
            else:
                Ls_norm += 0.5 * coef * delt
        kraus0_norm += Ls_norm

    total_norm.append(kraus0_norm)
    sqdelta = sp.sqrt(delt)
        
    for Ls in Lind.L_list:
        Ls_copy_2 = deepcopy(Ls)
        
        Ls_norm_2 = Ls_copy_2.pauli_norm()
        total_norm.append(Ls_norm_2 * sqdelta)
    ### Calculate the success prob here
    channel_sum_squares = 0.0
    for val in total_norm:
        channel_sum_squares += sp.simplify(val ** 2)
    success_prob_th = sp.simplify(1.0 / channel_sum_squares)
    return success_prob_th

def create_zerow_comp_state(r, h, p):
    """ Create the g state."""""
    logr = int(np.ceil(np.log2(r + 1)))
    greg = QuantumRegister(h * logr, 'g')
    sreg = QuantumRegister(h, 'res')
    g_vals = [bin(r)[2:]] * h
    statevecs = np.zeros(2 ** (h * logr), dtype = complex)
    qc = QuantumCircuit(greg)
    for i in range(h):
        combinations_list = list(combinations(range(r), i + 1))
        for comb in combinations_list:
            g_vals_cpy = deepcopy(g_vals)
            for j, val in enumerate(comb):
                if j == 0:
                    g_vals_cpy[-1] = bin(val)[2:].zfill(logr)
                else:
                    g_vals_cpy[-(j + 1)] = bin(val - comb[j - 1] - 1)[2:].zfill(logr)
            
            g_strings = ''.join(g_vals_cpy)
            int_ind = int(g_strings, 2)
            w = np.sqrt(p ** (r - i - 1) * (1 - p) ** (i + 1))
            statevecs[int_ind] = w

    end = 1
    for i in range(h):
        end += r * 2** (logr * i)
    
    ### The maximum index of register g is end - 1; 
    statevecs[end - 1] = np.sqrt(p ** r)
    norm = np.linalg.norm(statevecs)
    statevecs = Statevector(statevecs / norm)
    prep_gate = StatePreparation(statevecs)
    qc.append(prep_gate, qc.qubits)
    return statevecs, qc

def controls_on_r(qc_sub: QuantumCircuit, r: int):

    ### Identify value r
    logr = int(np.ceil(np.log2(r + 1)))
    bin_r = bin(r)[2:].zfill(logr)
    anc = QuantumRegister(1, 'a')
    ctrls = QuantumRegister(logr, 'c')
    sys_reg = QuantumRegister(qc_sub.num_qubits, 's')
    
    ## Controls, ancilla, system registers
    qc = QuantumCircuit(ctrls, anc, sys_reg)
    
    for idx, bit in enumerate(reversed(bin_r)):
        if bit == '0':
            qc.x(idx)

    qc.mcx(ctrls, anc[0])
    qc.x(anc[0])
    qc_sub_control = qc_sub.control(num_ctrl_qubits = 1, ctrl_state = '1')
    qc.compose(qc_sub_control, [anc[0]] + list(range(logr + 1, qc.num_qubits)), inplace=True)
    qc.x(anc[0])
    
    qc.mcx(ctrls, anc[0])
    for idx, bit in enumerate(reversed(bin_r)):
        if bit == '0':
            qc.x(idx)
 
    return qc

def phase_generator(targ, deg):
    """
    Generate phase factors for QSVT to a given matrix.
    """

    max_func = np.max(np.abs(targ(np.linspace(-1, 1, 10000))))
    # func = lambda s: targ(s) / max_func
    parity = deg % 2
    opts = {
        'intervals': [-1, 1],
        'objnorm': np.inf,
        'epsil': 1e-3,
        'npts': 1000,
        'fscale': 0.99,
        'isplot': False,
        'method': 'cvxpy',
        'maxiter': 100,
        'criteria': 1e-12,
        'useReal': True,
        'targetPre': True,
        'print': False
    }
    coef_full = cvx_poly_coef(targ, deg, opts)
  
    coef = coef_full[parity::2]

    opts['method'] = 'Newton'
    phi_proc, out = solve(coef, parity, opts)
    

   
#    ## Evaluate the approximation error
    x_list = np.linspace(-1, 1, 1000)

    func_cheby = lambda x: chebyshev_to_func(x, coef, parity, True)
    
    max_value = np.max(np.abs(func_cheby(x_list)))
    return phi_proc, max_value

def get_adaptive_qsp_phases(func, degree):
    poly = Chebyshev.interpolate(func, degree)
    coeffs = poly.coef.copy()
    x_list = np.linspace(-1, 1, 1000)
    func_cheby = lambda x: chebyshev_to_func(x, coeffs, degree % 2, True)
    max_value = np.max(np.abs(func_cheby(x_list)))
    if degree % 2 == 0:
        coeffs[1::2] = 0.0
    else:
        coeffs[0::2] = 0.0

    phases = QuantumSignalProcessingPhases(coeffs, signal_operator="Wx")
    return phases, max_value

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

def count_multiq_gates(qc: QuantumCircuit):
    count = 0
    countT = 0
    for gate, qargs, _ in qc.data:
        if len(qargs) > 1: 
            count += 1
        elif gate.name == 't' or gate.name == 'tdg':
            countT += 1
    return count, countT
def apply_poly_phases(phi_values, gadget, qc: QuantumCircuit, anc, ctrl):
    """
    Apply projection-controlled phase rotations to make the QSVT circuit.
    """
    qc.h(anc[0])
    for i, phi in enumerate(reversed(phi_values)):
        if ctrl is not None:
            qc.x(ctrl)
            qc.mcx(ctrl, anc[0])
        qc.rz(2 * phi, anc[0])
        if ctrl is not None: 
            qc.mcx(ctrl, anc[0])
            qc.x(ctrl)
        if i % 2 == 0 and i != len(phi_values) - 1:
            qc.append(gadget, qc.qubits[1:])
         
        elif i % 2 == 1 and i != len(phi_values) - 1:
            qc.append(gadget.inverse(), qc.qubits[1:])
    qc.h(anc[0])

    return qc
def qsvt_Hamiltonian(J: Matrixsum, t: float):
    """
    Create the QSVT circuit for the Hamiltonian terms e^-iJt
    J is the coherent term: J = H - 1/2i sum L^dag L
    q is number of quadrature points,
    and l is truncation order for term J.
    """
    ### Create the block-encoding of J
    qc_basic = block_encoding_matrixsum(J)
    subnorm_fac = sum(abs(coeff) for _, coeff in J.instances)
    sys_size = J.size
    ctrl_size = qc_basic.num_qubits - sys_size
    anc = QuantumRegister(1, 'a')
    if ctrl_size > 0:
        ctrl = QuantumRegister(ctrl_size, 'c')
    sys = QuantumRegister(sys_size, 's')

    ### Compute phase polynomials 
    deg = 4
    cos_func = lambda x: np.cos(subnorm_fac * t * x)
    cos_phi_values, max_value_cos = get_adaptive_qsp_phases(cos_func, deg)
    sin_func = lambda x: np.sin(subnorm_fac * t * x)
    sin_phi_values, max_value_sin = get_adaptive_qsp_phases(sin_func, deg - 1)
    
    cos_phi_values[0] += np.pi / 4 #type:ignore
    for i in range(1, len(cos_phi_values) - 1):
        cos_phi_values[i] -= np.pi / 2 #type: ignore
    cos_phi_values[-1] += np.pi / 4 #type: ignore

    sin_phi_values[0] += np.pi / 4 #type: ignore
    for i in range(1, len(sin_phi_values)):
        sin_phi_values[i] -= np.pi / 2 #type: ignore
    sin_phi_values[-1] +=  np.pi / 4 #type: ignore

    QSVT_basic_gadget = qc_basic.to_gate(label = "QSVT_basic_gadget") 
    
    if ctrl_size == 0:
        qc_sin = QuantumCircuit(anc, sys)
        qc_cos = QuantumCircuit(anc, sys)
        qc_sin = apply_poly_phases(sin_phi_values, QSVT_basic_gadget, qc_sin, anc, None)
        qc_cos = apply_poly_phases(cos_phi_values, QSVT_basic_gadget, qc_cos, anc, None)
    else:
        qc_sin = QuantumCircuit(anc, ctrl, sys)
        qc_cos = QuantumCircuit(anc, ctrl, sys)
        qc_sin = apply_poly_phases(sin_phi_values, QSVT_basic_gadget, qc_sin, anc, ctrl)
        qc_cos = apply_poly_phases(cos_phi_values, QSVT_basic_gadget, qc_cos, anc, ctrl)
    U_ctrl_cos = qc_cos.to_gate().control(1, ctrl_state = '0')
    U_ctrl_sin = qc_sin.to_gate().control(1, ctrl_state = '1')
  
    ## Prepare a LCU circuit for e^(iHt) = cos(Ht) - isin(Ht)
    ## An additional ancilla qubit for Hamiltonian evolution e^(-iHt), initialized in |+>
    sel = QuantumRegister(1, 'sel')
    if ctrl_size == 0:
        qc_main = QuantumCircuit(sel, anc, sys)
    else:
        qc_main = QuantumCircuit(sel, anc, ctrl, sys) 
    qc_main.ry(np.pi / 2, sel) # Prepare coeff (1/sqrt(2), 1/sqrt(2))
    qc_main.append(U_ctrl_cos, qc_main.qubits)
    qc_main.append(U_ctrl_sin, qc_main.qubits)
    qc_main.p(-np.pi / 2 ,sel)
    qc_main.ry(-np.pi / 2, sel)

    return qc_main, max_value_cos + max_value_sin

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
    prep_gate = StatePreparation(state_vector)
    qc.append(prep_gate, range(qubit_length))
    return qc

def exp_term_expansion(J: Matrixsum, K: int, t: float):
    J.mul_coeffs(-1j)
    sum_of_J = J.identity(J.size)
    for order in range(1, K + 1):
        if order == 1: 
            product_J = deepcopy(J)
            product_J.mul_coeffs(t)
            sum_of_J = sum_of_J.add(product_J)
            
        else: 
            product_J = matsum_mul(product_J, J)
            product_J.mul_coeffs(t / order)
            sum_of_J = sum_of_J.add(product_J)
    # succ_prob = sum_of_J.pauli_norm()
    return sum_of_J, sum_of_J.pauli_norm()

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

if __name__ == "__main__":
    
    H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    gamma = np.sqrt(0.1)/2 
    L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
        
    TFIM_lind = Lindbladian(H, L_list)
    success_prob_th = probs_from_lindblad(TFIM_lind)
    print("Theoretical success probability:", success_prob_th)
    r = 3
    equation = sp.Eq(success_prob_th, 4**(-1/r))
    initial_guess = 0.05
    delta_value = sp.nsolve(equation, sp.symbols('delta_t', real = True, positive = True), initial_guess)
    print("Solved delta_t value:", delta_value)
    print(delta_value * r)
    r, h = 1, 1
    g_state, qc_zerow = create_zerow_comp_state(r, h, 0.49)
    print(g_state, qc_zerow)
    exit(0)
    ccccx = XGate.control(self = XGate(), num_ctrl_qubits= 6, ctrl_state = '101000')
    qc = QuantumCircuit(3)
    qc.x(0)
    # qc.x(4)
    # qc.x(5)
    qc.save_statevector() #type: ignore
    print(qc.draw())
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    print(result.get_counts())
    statevec = result.get_statevector(compiled_circuit)
    print(statevec)

    # print(qc.draw())
    g_state, qc_zerow = create_zerow_comp_state(r, h, p)
