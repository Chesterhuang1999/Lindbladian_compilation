import numpy as np
from qsppack.utils import cvx_poly_coef
from qsppack.solver import solve
from qiskit.circuit.library import UnitaryGate, QFT, QFTGate, IntegerComparatorGate
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
import math 
from qsppack.utils import chebyshev_to_func, get_entry
import matplotlib.pyplot as plt
from copy import deepcopy
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli, Statevector, DensityMatrix
from itertools import combinations
from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector


# A = np.array([[1, 0,0, 0], [0, np.sqrt(1/3), -np.sqrt(2/3), 0], [0, np.sqrt(2/3), np.sqrt(1/3), 0], [0,0,0,1]])
# op = SparsePauliOp.from_operator(Operator(A))
# op = op.simplify(atol=1e-12)
# print(op.paulis, op.coeffs)
# for p, c in zip(op.paulis, op.coeffs):

#     print(isinstance(p, Pauli))
qc = QuantumCircuit(4)
qc.append(Pauli('XYZZ'), [0,1,2,3])
qc = qc.decompose()
qc_ctrl = qc.to_gate().control(3, ctrl_state='100')
qc = QuantumCircuit(7)
qc.append(qc_ctrl, range(7))
simulator = AerSimulator()
qc = transpile(qc, simulator, optimization_level=3)
print(qc.draw())
exit(0)

def create_zerow_comp_state(r, h, p):
    """ Create the g state."""""
    logr = int(np.ceil(np.log2(r)))
    greg = QuantumRegister(h * logr, 'g')
    sreg = QuantumRegister(h, 'res')
    g_vals = [bin(r)[2:]] * h
    statevecs = np.zeros(2 ** (h * logr), dtype = complex)
    qc = QuantumCircuit(greg, sreg)
    for i in range(h):
        combinations_list = list(combinations(range(r), i + 1))
        print(len(combinations_list))
        for comb in combinations_list:
            g_vals_cpy = deepcopy(g_vals)
            for j, val in enumerate(comb):
                if j == 0:
                    g_vals_cpy[-1] = bin(val)[2:].zfill(logr)
                else:
                    g_vals_cpy[-(j+ 1)] = bin(val - comb[j - 1] - 1)[2:].zfill(logr)

            g_strings = ''.join(g_vals_cpy)
            int_ind = int(g_strings, 2)
            w = p ** (r - i - 1) * (1 - p) ** (i + 1)
            statevecs[int_ind] = w
            
    statevecs[0] = p ** r
    norm = np.linalg.norm(statevecs) 
    statevecs = Statevector(statevecs / norm)
    qc.initialize(statevecs, greg)
    return statevecs, qc


r = 5
s = int(np.ceil(np.log2(r)))
h = 2
p = 0.1
create_zerow_comp_state(r, h, 0.9)
exit(0)
## Set maximal length to 1
 


dat = QuantumRegister(r, 'data')
res = QuantumRegister(s, 'res')
creg = ClassicalRegister(r + 1, 'cres')
anc = QuantumRegister(1, 'anc')
p = 0.1
angle = 2 * np.arcsin(np.sqrt(p))
qc = QuantumCircuit(dat, res, anc)
for i in range(r):
    qc.ry(angle, dat[i])
## Use phase estimation to count the Hamming weight
for j in range(s):
    qc.h(res[j])
    angle = 2 * np.pi * r /2**(s - j + 1)
    qc.rz(angle, res[j]) 
for j in range(s):
    angle = -2 * np.pi / (2 ** (s - j + 1))
    for i in range(r):
        qc.cx(res[j], dat[i])
        qc.rz(angle, dat[i])
        qc.cx(res[j], dat[i])

# inv_qft = QFTGate(s).inverse()
inverse_qft = QFT(s, inverse = True)
qc.compose(inverse_qft, res, inplace = True)
qc_temp = deepcopy(qc)
qc_to_gate_inv = qc.to_gate().inverse()
qc_to_gate = qc.to_gate()
## Initialize to |->

### Apply a Integer comparator
comparator = IntegerComparatorGate(s, 3)
qc.compose(comparator, res[:] + anc[:], inplace = True)



### Simulation
qc.add_bits(creg)
qc.measure(dat, creg[:-1])
qc.measure(anc, creg[-1])

N = 200000
simulator = AerSimulator()
ini_state = Statevector.from_label('0' * (qc.num_qubits))
qc = transpile(qc, simulator, optimization_level=1)

result = simulator.run(qc, shots = N, initial_state=ini_state).result()
counts = result.get_counts()
for key, value in counts.items():
    g = np.zeros(h, dtype = int)
    if key[0] == '0':
        w = list(key[1:])
        positions = [i for i, bit in enumerate(w) if bit == '1']
        
        for i in range(h):
            if i < len(positions):
                if i == 0:
                    g[i] = positions[i]
                else:
                    g[i] = positions[i] - positions[i - 1] - 1
            else:
                g[i] = r
    ## mapping: 
    # w = int(key, 2)
    # print(p**w * (1 - p)**(r - w) * math.comb(r, w))

exit(0)


mat = SparsePauliOp.from_list([('ZZIYX', 1)])
print(mat.dim)

qc = QuantumCircuit(5)
qc.append(mat, range(5))
qc = qc.decompose()
# qc = transpile(qc, basis_gates = ['u3', 'cx'], optimization_level=3)
print(qc.draw())
U_elem = qc.to_gate()

# for i in range(2**3):
#     ini_state = bin(i)[2:].zfill(3)
#     sv = Statevector.from_label(ini_state)
#     sv = sv.evolve(qc)
#     print(f'Input state: |{ini_state}>, output state: {sv}')

control_values = '00101'
qc = QuantumCircuit(10)

ctrl_mat = U_elem.control(len(control_values), ctrl_state=control_values)
qc.append(ctrl_mat, range(10))
qc = qc.decompose()
qc = transpile(qc, optimization_level = 3)
print(qc.draw())

mats_channel = [
        []
        for _ in range(3)
    ]
for i in range(3):
    for j in range(i + 2):
        mats_channel[i].append(Pauli('III'))
# print(mats_channel)


ctrl_size = 1
select_size = 1
sys_size = 2
vec_0 = Statevector.from_label('0' * ctrl_size)
proj_0 = vec_0.to_operator()
iden = Operator.from_label('I' * (select_size + sys_size))
proj_0 = proj_0.tensor(iden)

vec_0 = np.array([1,0]).T
vec_1 = np.array([np.sqrt(2)/2, np.sqrt(2)/2]).T
qc = QuantumCircuit(4)
qc.h(0)
qc.h(2)
initial_state = Statevector.from_label('0000')
final_state = initial_state.evolve(qc)
density = DensityMatrix(final_state)
# density = DensityMatrix(Statevector(np.kron(np.kron(vec_1, vec_0), np.kron(vec_1, vec_0))))
red_density = proj_0 @ density @ proj_0
print(density)
print(red_density)
print(DensityMatrix(red_density).trace())
# ## test benchmarkp
# beta = 100
# targ = lambda x: np.exp(-beta/2 * np.arcsin(x)**2)


# ## Gaussian weights (L^2 normalized)
# beta = 1.0
# sigma_e = 1 / beta 
# N = 200
# a = np.sqrt(2 * np.pi * N) * beta / 4
# coeff = np.pow(2/np.pi, 1/4) / np.sqrt(beta) ## Maximum value of Gaussian func
# scale = min(1, 1/coeff)
# # func = exp(-(2ax/N)^2/beta^2) x\in[-N/2, N/2, step = 1]
# # let s = sin(2x/N), then targ(s) = exp(-a^2 s^2 / beta^2)
# # a = \sqrt{2\pi N}\beta/4 
# targ = lambda s: np.exp(-a**2/ beta**2 * np.arcsin(s)**2)

# deg = 100
# parity = deg % 2

# opts = {
#     'intervals': [-np.sin(1), np.sin(1)],
#     'objnorm': np.inf,
#     'epsil': 1e-3,
#     'npts': 1000,
#     'fscale': 0.99,
#     'isplot': True,
#     'method': 'cvxpy',
#     'maxiter': 100,
#     'criteria': 1e-12,
#     'useReal': True,
#     'targetPre': True,
#     'print': True
# }

# coef_full = cvx_poly_coef(targ, deg, opts)
# coef = coef_full[parity::2]

# opts['method'] = 'Newton'

# phi_proc, out = solve(coef, parity, opts)




# ## Evaluate the approximation error
# xlist = np.linspace(-np.sin(1), np.sin(1), 1000)
# func = lambda x: chebyshev_to_func(x, coef, parity, True)
# targ_value = targ(xlist)
# func_value = func(xlist)
# QSP_value = get_entry(xlist, phi_proc, out)
# err = np.linalg.norm(QSP_value - func_value, np.inf)
# print('The residual error is')
# print(err)



# ## Evaluate QSPvalue
# print('Evaluating target function on real x values:')
# real_x_list = np.linspace(-N/2 , N/2 - 1, N, dtype = int)
# s_list = np.sin(2 * real_x_list / N)
# QSP_value_real = get_entry(s_list, phi_proc, out)
# func_value_real = targ(s_list)
# value_sum = np.sum(func_value_real**2)
# succ_prob = value_sum / N
# print(value_sum / N)
# # print(QSP_value_real)
# exit(0)

# plt.plot(xlist, QSP_value - func_value)
# plt.xlabel('$x$', fontsize=12)
# plt.ylabel('$g(x,\\Phi^*)-f_\\mathrm{poly}(x)$', fontsize=12)
# plt.show()


