import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import combinations
import sympy as sp
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp, Pauli
from qiskit.circuit.library import UnitaryGate, TGate, RZGate, HGate, C3XGate, StatePreparation, CHGate, XGate
from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector

from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry
from qsppack.solver import solve
import pennylane as qml
from channel_IR import * 
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
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

def phase_generator(targ, N, deg):
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
        'isplot': True,
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
#     func_value = func_cheby(x_list) * max_func
#     real_func_value = targ(x_list)
#     value_diff = func_value - real_func_value
#     QSP_value = get_entry(phi_proc, x_list, out)
#     err = np.linalg.norm(value_diff, np.inf)

#     real_x_list = np.linspace(-N/2, N/2 - 1, N, dtype = int)
#     s_list = np.sin(2 * real_x_list / N) 
#     func_value = func_cheby(s_list)
#     funcsum = np.sum(func_value**2)
#     succ_prob = funcsum / N 
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
