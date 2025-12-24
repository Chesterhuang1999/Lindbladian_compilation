import numpy as np
from qsppack.utils import cvx_poly_coef
from qsppack.solver import solve


from qsppack.utils import chebyshev_to_func, get_entry
import matplotlib.pyplot as plt

from qiskit.quantum_info import Operator, SparsePauliOp, Pauli


A = np.array([[1, 0,0, 0], [0, np.sqrt(1/3), -np.sqrt(2/3), 0], [0, np.sqrt(2/3), np.sqrt(1/3), 0], [0,0,0,1]])
op = SparsePauliOp.from_operator(Operator(A))
op = op.simplify(atol=1e-12)
print(op.paulis, op.coeffs)
for p, c in zip(op.paulis, op.coeffs):
    print(isinstance(p, Pauli))

# ## test benchmark
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


