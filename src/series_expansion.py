import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.library import save_statevector, save_density_matrix
from qutip import Qobj, tensor, basis
from channel_IR import *
from subroutine import *
from block_encoding import BlockEncoding
from baseline import simulate_lindblad 
from copy import deepcopy
import time
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm
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

#### Construct the Lindbladian evolution circuit via higher-order series expansion
def create_single_kraus(
        k: int, index: list, L_list: list, K1: int,
        t: float, ctrl_size: int, sys_size:int, eff_H: Matrixsum
        ):
    """ 
    Create the LCU encoding circuit by 
    Concatenating the block-encoding circuits for 
    each coherent gadget e^{J delta_t} and decay term L_j. 
    A_j = sqrt(w_(jk) * ... * w_(jk,...j1)) e^J(t - x_(jk))L_(lk)e^J(x_(jk)- x_(jk,j_(k-1)))..... L_(l1)e^J(x_(j_k, j_(k-1),...,j_1))

    Parameters: 
    k (int): expansion order
    q (int): number of quadrature points
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

    ## Generate quadrature points and weights according to a biased Legendre function
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
            delta_t = points[k - j - 1] - points[k - j] 

        ### Concatenate the block-encoding for coherent term e^{J delta_t} = sum_l=0^K1 (J^l t^l/(l!)) 
        qc_C, coeff_C = construct_circuit_coherent(eff_H, K1, delta_t)
        qc_elem.compose(qc_C, qubits = qc_elem.qubits[ctrl_offset: ctrl_offset + ctrl_reg_size[2 * j]] 
                        + qc_elem.qubits[total_ctrl_size:], inplace = True)
        ctrl_offset += ctrl_reg_size[2 * j]
        coeff_sum_index *= coeff_C

        ### Concatenate the block-encoding for incoherent term L_(l_(k-j))
        if j < k:
            L_j = L_list[l_list[k - j - 1]]
            coeff_sum_index *= L_j.pauli_norm() ## sum of coefficients 
            qc_L = BlockEncoding(L_j).circuit()
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

def construct_circuit_coherent(J: Matrixsum, K1: int, t: float):
    J.mul_coeffs(-1j)
    sum_of_J = J.identity(J.size)
    for order in range(1, K1 + 1):
        if order == 1: 
            product_J = deepcopy(J)
            product_J.mul_coeffs(t)
            sum_of_J = sum_of_J.add(product_J)
            
        else: 
            product_J = matsum_mul(product_J, J)
            product_J.mul_coeffs(t / order)
            sum_of_J = sum_of_J.add(product_J)

    # qc_J = block_encoding_matrixsum(sum_of_J)
    qc_J = BlockEncoding(sum_of_J).circuit()
    succ_prob = sum_of_J.pauli_norm()
    return qc_J, succ_prob


def higher_order_Lind_expansion(Lind: Lindbladian, K: int, q: int, t: float, ini_state, K1: int):
    """
    Create the higher-order Lindblad evolution expansion circuit
    using Kth order series expansion with q quadrature points.
    """
    ### Zeroth order term: e^Jt
    ### Here we just compute H_eff, and J = -iH_eff is incorporated into LCU construction
    effective_H = Lind.effective_H()
    m = len(Lind.L_list)
    sys_size = effective_H.size
    sum_of_kraus = 0
    qc_ensemble = []
    coeff_sum_total = []
    ctrl_sizes = []
    ctrl_size_max = 0
    kraus_count = 1

    ### Construct the circuit for A_0 = e^{Jt}
    qc_info = construct_circuit_coherent(effective_H, K1, t)
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
            qc_elem, coeff_sum_index, elem_ctrl_size = create_single_kraus(k, index, Lind.L_list, K1, t, 
                                                    qc_coh_ctrl_size, sys_size, effective_H)
            ctrl_size_max = max(elem_ctrl_size, ctrl_size_max)
            qc_ensemble.append(qc_elem)
            coeff_sum_total.append(coeff_sum_index)
            ctrl_sizes.append(elem_ctrl_size)
    print(f"Total Kraus operators: {kraus_count}")

    #Define the registers
    sel_size = int(np.ceil(np.log2(kraus_count))) 
    reg_sizes = [sel_size, ctrl_size_max, sys_size]
    select_reg = QuantumRegister(sel_size, 'sel')
    control_reg = QuantumRegister(ctrl_size_max, 'ctrl')
    system_reg = QuantumRegister(sys_size, 'sys')

    ## Prepare superposition over selection register
    coeff_sum_square = sum([abs(c)**2 for c in coeff_sum_total])
    norm_coeffs = [abs(c)**2 / coeff_sum_square for c in coeff_sum_total]
    weights = np.zeros(2**sel_size, dtype = float)
    sel_state = np.zeros(2**sel_size, dtype = float)
    for i, nc in enumerate(norm_coeffs):
        weights[i] = nc
        sel_state[i] = np.sqrt(nc)
    # qc_prep = lcu_prepare_tree(weights)
    # qc_main.compose(qc_prep, qubits = qc_main.qubits[:sel_size], inplace = True) #type : ignore
    qc_main = QuantumCircuit(select_reg, control_reg, system_reg)
    state_sys_ctrl = Statevector.from_label(ini_state + '0' * ctrl_size_max)
    state_tot = state_sys_ctrl.tensor(Statevector(sel_state))
    qc_main.initialize(state_tot)
    print("Selection register prepared")

    ### Append A_0 = e^{Jt} circuit
    qc_coh_control = qc_info[0].to_gate().control(num_ctrl_qubits = sel_size, ctrl_state = '0' * sel_size)
    qc_main.append(qc_coh_control, qargs = qc_main.qubits[:sel_size] + qc_main.qubits[sel_size:sel_size + qc_coh_ctrl_size] 
                    + qc_main.qubits[sel_size + ctrl_size_max:])
    print("A_0 construct complete")

    ### Append other Kraus operator circuit
    for ind in range(1, kraus_count):
        qc_elem = qc_ensemble[ind -1]
        ctrl_size = ctrl_sizes[ind - 1]
        select_value = bin(ind)[2:].zfill(sel_size)
        qc_elem_control = qc_elem.to_gate().control(num_ctrl_qubits = sel_size, ctrl_state = select_value)
        qc_main.append(qc_elem_control, qargs = qc_main.qubits[:sel_size] + qc_main.qubits[sel_size: sel_size + ctrl_size]
                        + qc_main.qubits[sel_size + ctrl_size_max:])
    print("All Kraus operators constructed")

    return qc_main, reg_sizes, coeff_sum_square

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

def projection_vec(sv: Statevector, dim: int, anc_size: int, ctrls: list):
    sel_size = anc_size - len(ctrls)
    print(dim, anc_size, sel_size)
    proj_0 = Operator.from_label('0' * len(ctrls))
    if sel_size == 0:
        iden = Operator.from_label('I' * (dim - anc_size))
        proj_full = iden.tensor(proj_0)
        projected_vec = np.dot(proj_full, sv)
        projected_vec = projected_vec[::2**anc_size]
        return Statevector(zero_small_complex(projected_vec, tol = 1e-6))
    else:
        iden_sel = Operator.from_label('I' * (sel_size))
        iden_sys = Operator.from_label('I' * (dim - anc_size))
        proj_full = iden_sys.tensor(proj_0).tensor(iden_sel) 
        projected_vec = np.dot(proj_full, sv)
        projected_dm_sys = partial_trace(DensityMatrix(projected_vec), list(range(anc_size)))
        return DensityMatrix(zero_small_complex(np.asarray(projected_dm_sys), tol = 1e-6))


def simulate_circuit_statevec(qc: QuantumCircuit, reg_sizes = [0, 0, 0]):
    simulator = AerSimulator(method = 'statevector')
    ancilla_regs = []
    for qbit in qc.qubits:
        c = qbit._register
        if c.name != 's':
            ancilla_regs.append(qbit)
    if reg_sizes == [0, 0, 0]:
        sel_size = 0
        ctrl_size = len(ancilla_regs)
        sys_size = qc.num_qubits - ctrl_size
    else:
        sel_size, ctrl_size, sys_size = reg_sizes

    qc.save_statevector(label = 'final_state')
    qc = transpile(qc, simulator, optimization_level=2)
    
    ### Sim task I: Get final statevector
    
    result = simulator.run(qc, shots = 1).result()

    final_state = result.data()['final_state']
    
    final_state_sys = projection_vec(final_state, sel_size + ctrl_size + sys_size, sel_size + ctrl_size, list(range(sel_size, sel_size + ctrl_size)))
    

    return final_state_sys
def simulate_circuit_TN(qc: QuantumCircuit, ini_state: Statevector, reg_sizes: list, return_opt: str = 'full'):
    """
    Construct a tensor network simulation of the circuit qc with initial state ini_state.
    """
    sel_size, ctrl_size, sys_size = reg_sizes
    ancilla_size = sel_size + ctrl_size
    simulator = AerSimulator(method = 'matrix_product_state')
    simulator.set_options(matrix_product_state_max_bond_dimension=64)
    qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
    qc_sim.initialize(ini_state)
    qc_sim.compose(qc, qc.qubits, qc.clbits, inplace = True)
    
    qc_sim = transpile(qc_sim, simulator, optimization_level = 1)
    
    if return_opt == 'full':
        qc_sim.save_density_matrix(qubits = list(range(qc_sim.num_qubits)), label = 'final_dm') # type: ignore
        result = simulator.run(qc_sim, shots = 1).result()
        raw_rho = result.data()['final_dm']
        final_rho = DensityMatrix(normalize(projection_op(raw_rho, qc_sim.num_qubits, ctrl_size))) #type: ignore
        final_rho_sys = np.asarray(partial_trace(final_rho, list(range(sel_size))))
        return final_rho_sys
    else:
        anc_output = ClassicalRegister(ctrl_size, 'c_out')
        qc_sim.add_register(anc_output)
        for i in range(ctrl_size):
            qc_sim.measure(qc_sim.qubits[i + sel_size], anc_output[i])
    
        with qc_sim.if_test((anc_output, 0)):
            qc_sim.save_density_matrix(qubits = list(range(ancilla_size, ancilla_size + sys_size)), label = 'final_sys_dm') # type: ignore
            
        repeat = 100
        for i in range(repeat):
            result = simulator.run(qc_sim, shots = 1).result()
            print(result.get_counts())
            if list(result.get_counts().keys())[0] == '0'*ctrl_size:
                raw_rho = result.data()['final_sys_dm']
                break
        # final_rho_sys = np.asarray(partial_trace(raw_rho, list(range(sel_size))))
        final_rho_sys = np.asarray(raw_rho)
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
    K = 2
    q = 4
    t = 0.1
    test_case = 3
    H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    gamma = np.sqrt(0.1)/2 
    L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    delta_t = 0.1
    TFIM_lind = Lindbladian(H, L_list)
  
    if test_case == 2:
        H = [('Z', 1j)]
        L = []
        TFIM_lind = Lindbladian(H, L)
        H_eff = TFIM_lind.effective_H()
        print(H_eff)
        qc, max_value = qsvt_Hamiltonian(H_eff, delta_t)
        qc_n = QuantumCircuit(qc.num_qubits)

        start = time.time()
        H_evo = Operator(expm(-1j * H_eff.eff_op() * delta_t)) #type: ignore
        ini_state_qsvt = Statevector.from_label('+' + '0' * (qc.num_qubits - 1)) 
        ini_state_baseline = Statevector.from_label('+')
        final_state_baseline = normalize(ini_state_baseline.evolve(H_evo))
        print("Baseline final state:")
        print(final_state_baseline)
        qc_n.initialize(ini_state_qsvt)
        qc_n.compose(qc, qc_n.qubits, inplace = True)
    
        final_state_sys = simulate_circuit_statevec(qc_n, reg_sizes = [0, qc.num_qubits - 1, 1])
        print("QSVT final state:")
        print(final_state_sys)
        end = time.time()
        print(f"QSVT simulation time: {end - start} seconds")
        
        diff = DensityMatrix(final_state_sys) - DensityMatrix(final_state_baseline)
        err = np.linalg.norm(diff, ord = 'nuc') / 2
        print(f"Simulate error: {err}")
    
    elif test_case == 3:
        # weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0]
        # statevec = Statevector([np.sqrt(w) for w in weights])
        # qc_prep = lcu_prepare_tree(weights)
        # qc = QuantumCircuit(7)
        # qc.compose(qc_prep, qc.qubits[:4], inplace = True)
        # print(qc.draw())
        # ini_state = Statevector.from_label('0000000')
        # print(ini_state.evolve(qc))

        # qc = QuantumCircuit(7)
        # state1 = Statevector.from_label('000')
        # state1 = statevec.tensor(state1)
        # ini_state = Statevector.from_label('0000000')
        # qc.append(StatePreparation(state1), qc.qubits)
        # state2 = ini_state.evolve(qc)
        # err = np.linalg.norm(DensityMatrix(state1) - DensityMatrix(state2), ord = 'nuc') / 2
        # print(f"Simulate error: {err}")
       

        H = [('I', 1)]
        v = 0.5
        t = 0.1
        K1 = 5
        L_list = [[('X', np.sqrt(v + 1)), ('Y', 1j * np.sqrt(v + 1))], [('X', np.sqrt(v)), ('Y', -1j * np.sqrt(v))]]
        decay_lind = Lindbladian(H, L_list)
        
   
        ini_state = Statevector.from_label('+')
        H_qobj, L_qobj_list = construct_qobj_lind(decay_lind, decay_lind.H.size)
        qplus = (basis(2,0) + basis(2,1)).unit()
        ini_state_qobj = qplus
        final_dens = simulate_lindblad(H_qobj, L_qobj_list, ini_state_qobj, t, r = 10)
        print("Baseline final density:", final_dens)
        
        qc, reg_sizes, coeff_sum_sq = higher_order_Lind_expansion(decay_lind, K, q, t, '+', K1)
        print(reg_sizes)
        qc_n = QuantumCircuit(qc.num_qubits)
        ini_state_sim = Statevector.from_label('+' + '0' * (qc.num_qubits - 1))
        qc_n.initialize(ini_state_sim)
        qc_n.compose(qc, qc_n.qubits, inplace = True)

        final_state_sys = simulate_circuit_statevec(qc_n, reg_sizes = reg_sizes)
        final_state_sys = coeff_sum_sq * final_state_sys 
        # final_dens_sys = simulate_circuit_TN(qc, ini_state, reg_sizes, 'measure')
        
        # print("Simulated final density:", DensityMatrix(final_dens_sys))
    
        
        diff = final_state_sys - final_dens
        err = np.linalg.norm(diff, ord = 'nuc') / 2
        print(f"Simulate error: {err}")
        