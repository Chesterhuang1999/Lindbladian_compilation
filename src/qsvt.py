###########################
#   Implementation of QSVT algorithm. #
#   We focus on a specific use case:  #
#  approximating a function f(x) on a superposition state |x>. #
###########################


import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
import math
from qsppack.utils import cvx_poly_coef, chebyshev_to_func, get_entry
import matplotlib.pyplot as plt
from qsppack.solver import solve

def chebyshev_fit(func, degree, a, N):
    """
    Fit a function using Chebyshev polynomials up to a given degree.
    
    Args:
        func (callable): The function to approximate.
        degree (int): The degree of the Chebyshev polynomial.
        N: (int): Number of sample points, should be even, 2^k is recommended.
    Returns:
        list: Coefficients of the Chebyshev polynomial.
    """
    # Sample points
    # intervals = np.linspace(- N / 2, N / 2 - 1, N, dtype = int)
    # x = np.array([np.sin(2 * k / N)/ np.sin(1) for k in intervals])
    M = 800
    x = np.cos((2 * np.arange(M) + 1) * np.pi / (2 * M))
    y = func(a * np.arcsin(np.sin(1) * x))
    # Compute coefficients
    coeffs = np.polynomial.chebyshev.chebfit(x, y, degree)
    # Construct the polynomial 
    poly = np.polynomial.chebyshev.Chebyshev(coeffs, domain=[-1, 1])
    xs = np.linspace(-1, 1, 1000)
    ys = poly(xs)
    yt = func(a * np.arcsin(np.sin(1) * xs))
    max_err = np.max(np.abs(ys - yt))
    print(f"Max approximation error: {max_err}")
    return coeffs


## Example usage
# beta = 1.0 # inverse temperature
# func = lambda x: np.pow(1/(2 *np.pi), 1/2) * np.exp(-x**2/2)
# # func = lambda x: np.pow(2/np.pi, 1/4) * np.exp(-x**2/(beta **2))/ np.sqrt(beta)
# print(chebyshev_fit(func, 8, 1, 100))


## qsppack usage 
def phase_generator(func, beta, N, deg):
    """
    Generate phase factors for QSVT to a given matrix.
    """
    sigma_e = 1 / beta 
    N = 200
    a = np.sqrt(2 * np.pi * N) * beta / 4
    coeff = np.pow(2/np.pi, 1/4) / np.sqrt(beta) ## Maximum value of Gaussian func
    scale = min(1, 1/coeff)
    a = np.sqrt(2 * np.pi * N) * beta /4

    targ = lambda s: np.exp(-a**2/ beta**2 * np.arcsin(s)**2)

    max_func = np.max(np.abs(targ(np.linspace(-1, 1, 10000))))
    func = lambda s: targ(s) / max_func
    print(max_func)
    parity = deg % 2
    opts = {
        'intervals': [-np.sin(1), np.sin(1)],
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
        'print': True
    }
    coef_full = cvx_poly_coef(func, deg, opts)
    coef = coef_full[parity::2]

    opts['method'] = 'Newton'
    phi_proc, out = solve(coef, parity, opts)
    
    ## Evaluate the approximation error
    x_list = np.linspace(-np.sin(1), np.sin(1), 1000)

    func_cheby = lambda x: chebyshev_to_func(x, coef, parity, True)

    func_value = func_cheby(x_list) * max_func
    real_func_value = targ(x_list)
    value_diff = func_value - real_func_value
    QSP_value = get_entry(phi_proc, x_list, out)
    err = np.linalg.norm(value_diff, np.inf)

    real_x_list = np.linspace(-N/2, N/2 - 1, N, dtype = int)
    s_list = np.sin(2 * real_x_list / N) 
    func_value = func_cheby(s_list)
    funcsum = np.sum(func_value**2)
    succ_prob = funcsum / N 


    return phi_proc, succ_prob

if __name__ == "__main__":
    func = lambda x: np.pow(4/np.pi, 1/4) * np.exp(-2*x**2)

    phi_values, succ_prob = phase_generator(func, np.sqrt(2), 32, 64)
    print(succ_prob)
