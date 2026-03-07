from series_expansion import *
from channel_IR import Lindbladian

from channel_IR import *
from qutip import qeye, sigmax, sigmaz, sigmam, tensor
# ===== Test III.2 extension: TFIM + decay model (follow Cell 13 simulation pipeline) =====

def build_tfim_decay_lindbladian_pauli(N: int, Delta: float, J: float, gamma: float):
    """
    Build the model in Pauli-label form consistent with:
      H = Delta * sum_j Z_j - J * sum_j X_j X_{j+1}
      L_j = sqrt(gamma) * sigma^-_j,  j = 1..N-1 (following the provided QuTiP loop range)

    sigma^- = (X - iY)/2, so each L_j is represented by two Pauli terms.
    """
    H_terms = []
    L_terms = []

    # Hamiltonian: Delta * sum_j Z_j
    for j in range(N):
        z_str = ''.join('Z' if k == j else 'I' for k in range(N))
        H_terms.append((z_str, Delta))

    # Hamiltonian: -J * sum_j X_j X_{j+1}, open boundary (j=0..N-2)
    for j in range(N - 1):
        xx_str = ''.join('X' if (k == j or k == j + 1) else 'I' for k in range(N))
        H_terms.append((xx_str, -J))

    # Lindblad jumps: sqrt(gamma) * sigma^-_j = sqrt(gamma)/2 * (X_j - iY_j), j=0..N-2
    amp = np.sqrt(gamma) / 2
    for j in range(N):
        x_str = ''.join('X' if k == j else 'I' for k in range(N))
        y_str = ''.join('Y' if k == j else 'I' for k in range(N))
        L_terms.append([(x_str, amp), (y_str, -1j * amp)])

    return H_terms, L_terms


def build_tfim_decay_qutip_reference(N: int, Delta: float, J: float, gamma: float):
    """Construct QuTiP H and L_list matching the user-provided model exactly."""
    sz = sigmaz()
    sx = sigmax()

    H_qobj = 0 * tensor([qeye(2) for _ in range(N)]) #type: ignore

    # Delta * sum_j Z_j
    for j in range(1, N + 1):
        left = [qeye(2) for _ in range(j - 1)]
        right = [qeye(2) for _ in range(N - j)]
        H_qobj += Delta * tensor(left + [sz] + right) #type: ignore

    # -J * sum_j X_j X_{j+1}
    for j in range(1, N):
        left = [qeye(2) for _ in range(j - 1)]
        right = [qeye(2) for _ in range(N - j - 1)]
        H_qobj += (-J) * tensor(left + [sx, sx] + right) #type: ignore

    L_qobj = []
    # j in range(1, N) exactly as provided
    for j in range(N):
        left = [qeye(2) for _ in range(j)]
        right = [qeye(2) for _ in range(N - j - 1)]
        L_qobj.append(np.sqrt(gamma) * tensor(left + [sigmam()] + right)) #type: ignore

    return H_qobj, L_qobj

# ===== Parameters =====
N_list = [2, 4, 6]
K_list = [1]
K1 = 3
q = 3

Delta = 1.0
J = 1.0
gamma = 0.1

t = 0.1
repeat = 1
r_solver = 6

n_qubits = 2
H_terms, L_terms = build_tfim_decay_lindbladian_pauli(n_qubits, Delta, J, gamma)
H = []
v = 0.5
scaling = 1.0
L_list = [[('X', np.sqrt((v + 1))/scaling), ('Y', 1j * np.sqrt((v + 1))/scaling)], [('X', np.sqrt( v)/scaling), ('Y', -1j * np.sqrt(v)/scaling)]]
decay_lind = Lindbladian(H, L_list)
# H_qobj, L_qobj_list = build_tfim_decay_qutip_reference(n_qubits, Delta, J, gamma)
TFIM_lind = Lindbladian(H_terms, L_terms)
H_eff = decay_lind.effective_H()

qc_no, succ_prob_no = construct_circuit_coherent(H_eff, K1, t, opt = 'No')
qc_opt, succ_prob_opt = construct_circuit_coherent(H_eff, K1, t, opt = 'Matrix-order')
simulator = AerSimulator(method='statevector')
print(qc_opt.draw())

ini_sys = Statevector.from_label('+' * n_qubits)
w = qc_no.num_qubits - n_qubits


sel_state = [1.0]
final_no = simulate_circuit_statevec(qc_no, ini_sys, sel_state=sel_state, reg_sizes=[0, w, n_qubits])
final_opt = simulate_circuit_statevec(qc_opt, ini_sys, sel_state=sel_state, reg_sizes=[0, w, n_qubits])
print(final_no, final_opt)
# final_no = final_no / np.trace(final_no)
# final_opt = final_opt / np.trace(final_opt)
diff = final_no - final_opt
err = np.linalg.norm(diff) 
print(f"Error between No opt and Matrix-order opt: {err}")
