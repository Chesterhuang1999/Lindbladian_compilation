import numpy as np
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, basis, Qobj
from qutip import mesolve, sesolve
from channel_IR import Matrixsum, Lindbladian
from copy import deepcopy
def approx_state(sv: np.ndarray, tol: float = 1e-8, rnd = 4):
    """
    Approximate a vector/matrix by removing small amplitudes
    """
    
    sv_array = deepcopy(sv)
    sv_array[np.abs(sv_array) < tol] = 0.
    # sv_array.imag[np.abs(sv_array.imag) < tol] = 0. 
    sv_array = np.round(sv_array, rnd)
    return sv_array

def simulate_Hamiltonian(H: Matrixsum, x: int, psi_0, duration: float = 1.0):
    H_eff = H.eff_op()
    N = int(np.log2(H_eff.dim[0])) #type: ignore
    H_qobj = Qobj(H.eff_op(), dims = [[2]*N, [2]*N])
    
    
    tlist = np.linspace(0, duration, 10)
    result = sesolve(H_qobj, psi_0, tlist)
    
    return result.states[x]
def simulate_lindblad(H: Qobj, L_list: list, psi_0: Qobj, duration: float = 1.0, r: int = 10):
    """
    Simulate the Lindblad evolution using QuTiP mesolve
    """


    # psi_0 = tensor(basis(2,0) for _ in range(N))  # type: ignore

    tlist = np.linspace(0, duration, r)

    result = mesolve(H, psi_0, tlist, c_ops = L_list, e_ops = [])
    final_state_qobj = result.states[-1].data_as("ndarray")
    
    return approx_state(final_state_qobj, tol = 1e-8, rnd = 6)

if __name__ == "__main__":
    H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    gamma = np.sqrt(0.1)/2 
    L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    delta_t = 0.1
    TFIM_lind = Lindbladian(H, L_list)
    H_eff = TFIM_lind.effective_H()
    psi_plus = (basis(2,0) + basis(2,1)).unit()
    psi_0 = tensor(psi_plus, psi_plus, basis(2, 0))
    # psi_0 = tensor(basis(2,0) for _ in range(3))  # type: ignore
    print(psi_0)
    
    # state_qutip = simulate_Hamiltonian(H_eff, 1, psi_0, duration = 1.0)

    # print(state_qutip)    



