import numpy as np
import time
from itertools import product

from channel_IR import *
from block_encoding import BlockEncoding 
from subroutine import get_adaptive_qsp_phases, apply_poly_phases
from scipy.linalg import expm, cosm, sinm

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp, random_pauli_list, Operator, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis.evolution import QDrift


### qsvt algorithm for Hamiltonian simulation exp(-iHt) using block-encoding of H. 
def qsvt_Hamiltonian(J: Matrixsum, t: float, deg: int = 4, opt = 'No'):
    """
    Create the QSVT circuit for the Hamiltonian terms e^-iJt
    J is the coherent term: J = H - 1/2i sum L^dag L
    q is number of quadrature points,
    and l is truncation order for term J.
    """
    ### Create the block-encoding of J
    # qc_basic = block_encoding_matrixsum(J)
    be = BlockEncoding(J)
    
    qc_basic = be.circuit(opt = opt)
    # print(gate_count_summary(qc_basic, basis_gates = ['cx', 'u3'], optimization_level= 1))
    subnorm_fac = sum(abs(coeff) for _, coeff in J.instances)
    sys_size = J.size
    ctrl_size = qc_basic.num_qubits - sys_size
    anc = QuantumRegister(1, 'a')
    if ctrl_size > 0:
        ctrl = QuantumRegister(ctrl_size, 'c')
    sys = QuantumRegister(sys_size, 's')

    ### Compute phase polynomials 
    cos_func = lambda x: np.cos(subnorm_fac * t * x)
    cos_phi_values, max_value_cos = get_adaptive_qsp_phases(cos_func, deg)
    sin_func = lambda x: np.sin(subnorm_fac * t * x)
    sin_phi_values, max_value_sin = get_adaptive_qsp_phases(sin_func, deg - 1)
    
    # cos_phi_values[0] += np.pi / 4 #type:ignore
    # for i in range(1, len(cos_phi_values) - 1):
    #     cos_phi_values[i] -= np.pi / 2 #type: ignore
    # cos_phi_values[-1] += np.pi / 4 #type: ignore

    # sin_phi_values[0] += np.pi / 4 #type: ignore
    # for i in range(1, len(sin_phi_values)):
    #     sin_phi_values[i] -= np.pi / 2 #type: ignore
    # sin_phi_values[-1] +=  np.pi / 4 #type: ignore

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


    return qc_basic, qc_cos, max_value_cos + max_value_sin
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
 
    return qc_basic, qc_main, max_value_cos + max_value_sin


def matrixsum_to_sparse_pauli(H: Matrixsum, atol: float = 1e-12) -> SparsePauliOp:
    """Convert Matrixsum(PauliAtom) to SparsePauliOp for evolution synthesis."""
    pauli_terms = []
    for atom, coeff in H.instances:
        if not isinstance(atom, PauliAtom):
            raise TypeError("QDrift baseline currently supports Matrixsum with PauliAtom terms only.")
        c = complex(coeff) * complex(atom.phase)
        if abs(c) > atol:
            pauli_terms.append((atom.expr, c))

    if len(pauli_terms) == 0:
        return SparsePauliOp.from_list([("I" * H.size, 0.0)])

    return SparsePauliOp.from_list(pauli_terms).simplify(atol=atol)


def qdrift_hamiltonian(
    H: Matrixsum | SparsePauliOp | list[tuple[str, complex]],
    t: float,
    reps: int = 1,
    seed: int | None = None,
    basis_gates: list[str] | None = None,
    optimization_level: int = 1,
) -> QuantumCircuit:
    """
    Build a QDrift circuit U ~ exp(-i H t).
    H can be Matrixsum / SparsePauliOp / list[(pauli_label, coeff)].
    """
    if isinstance(H, Matrixsum):
        H_op = matrixsum_to_sparse_pauli(H)
    elif isinstance(H, SparsePauliOp):
        H_op = H.simplify()
    else:
        H_op = SparsePauliOp.from_list([(p, complex(c)) for p, c in H]).simplify()

    try:
        synth = QDrift(reps=reps, seed=seed)
    except TypeError:
        synth = QDrift(reps=reps)

    evo = PauliEvolutionGate(H_op, time=t, synthesis=synth)
    n_qubits = H_op.num_qubits
    if n_qubits is None:
        raise ValueError("Unable to infer qubit count from Hamiltonian.")
    n_qubits = int(n_qubits)
    qc = QuantumCircuit(n_qubits, name="qdrift_evo")
    qc.append(evo, range(n_qubits))
    qc = qc.decompose(reps=2)

    if basis_gates is not None:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)
    return qc


def gate_count_summary(
    qc: QuantumCircuit,
    basis_gates: list[str] | None = None,
    optimization_level: int = 1,
) -> dict:
    tqc = transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)
    ops_raw = tqc.count_ops()
    ops = {str(k): int(v) for k, v in ops_raw.items()}
    return {
        "num_qubits": tqc.num_qubits,
        "depth": tqc.depth(),
        "size": tqc.size(),
        "cx": int(ops.get("cx", 0)),
        # "t": int(ops.get("t", 0) + ops.get("tdg", 0)),
        "u3": int(ops.get("u3", 0)),
        "ops": ops,
    }


def hamiltonian_l1_norm(
    H: Matrixsum | SparsePauliOp | list[tuple[str, complex]],
    atol: float = 1e-12,
) -> float:
    """Compute lambda = sum_j |h_j| for a Pauli-decomposed Hamiltonian."""
    if isinstance(H, Matrixsum):
        lam = 0.0
        for atom, coeff in H.instances:
            if not isinstance(atom, PauliAtom):
                raise TypeError("QDrift bound currently supports Matrixsum with PauliAtom terms only.")
            c = complex(coeff) * complex(atom.phase)
            if abs(c) > atol:
                lam += abs(c)
        return float(lam)

    if isinstance(H, SparsePauliOp):
        H_simple = H.simplify(atol=atol)
        return float(sum(abs(complex(c)) for c in H_simple.coeffs if abs(c) > atol))

    return float(sum(abs(complex(c)) for _, c in H if abs(c) > atol))


def qdrift_reps_lower_bound(
    H: Matrixsum | SparsePauliOp | list[tuple[str, complex]],
    t: float = 0.1,
    epsilon: float = 1e-3,
    prefactor: float = 2.0,
    atol: float = 1e-12,
) -> dict:
    """
    Conservative QDrift repetition lower bound from
    error <= prefactor * (lambda * t)^2 / r.
    Returns both exact real-valued bound and integer lower bound.
    """
    if t < 0:
        raise ValueError("t must be non-negative.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if prefactor <= 0:
        raise ValueError("prefactor must be positive.")

    lam = hamiltonian_l1_norm(H, atol=atol)
    r_real = prefactor * (lam * t) ** 2 / epsilon
    r_int = int(np.ceil(r_real))

    return {
        "lambda": float(lam),
        "t": float(t),
        "epsilon": float(epsilon),
        "prefactor": float(prefactor),
        "r_lower_bound_real": float(r_real),
        "r_lower_bound_int": r_int,
    }

if __name__ == "__main__":
    from simulator_utils import simulate_circuit_statevec, normalize

    delta_t = 0.1
    err_th = 1e-4
    pauli_2bit = [''.join(p) for p in product('IXYZ', repeat=2)]

    total_cases = 0
    erroneous_cases = 0

    for P1 in pauli_2bit:
        for P2 in pauli_2bit:
            H = [(P1, -1), (P2, -1)]
            L_list = []
            lind = Lindbladian(H, L_list)
            H_eff = lind.H

            try:
                _, qc, _ = qsvt_Hamiltonian(H_eff, delta_t, deg=4, opt='Matrix-order')
                H_evo = Operator(expm(-1j * H_eff.eff_op() * delta_t)) #type: ignore
                # H_cos = Operator(cosm(H_eff.eff_op() * delta_t)) #type: ignore
            except Exception as exc:
                total_cases += 1
                erroneous_cases += 1
                print(f"Erroneous case: P1={P1}, P2={P2}, build_error={type(exc).__name__}: {exc}")
                continue

            case_bad = False
            for i in range(4):
                start = time.time()
                bin_str = bin(i)[2:].zfill(2)
                ini_state_qsvt = Statevector.from_label(bin_str + '0' * (qc.num_qubits - 2))
                ini_state_baseline = Statevector.from_label(bin_str)
                final_state_baseline = normalize(ini_state_baseline.evolve(H_evo))
                final_state_sys = simulate_circuit_statevec(
                    qc,
                    ini_state_qsvt,
                    None,
                    reg_sizes=[0, qc.num_qubits - 2, 2],
                )
                _ = time.time() - start

                diff = DensityMatrix(final_state_sys) - DensityMatrix(final_state_baseline)
                err = np.linalg.norm(diff, ord='nuc') / 2
                if err > err_th:
                    case_bad = True
                    print(f"Erroneous case: P1={P1}, P2={P2}, err={err:.6e}")
                    break

            total_cases += 1
            if case_bad:
                erroneous_cases += 1

    print(f"Checked {total_cases} cases over all 2-bit Pauli pairs.")
    print(f"Erroneous cases (err > {err_th}): {erroneous_cases}")

