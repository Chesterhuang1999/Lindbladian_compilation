import numpy as np
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Kraus, DensityMatrix, Stinespring
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Isometry

try:
    from .block_encoding import BlockEncoding
except ImportError:
    from block_encoding import BlockEncoding

try:
    from .channel_IR import Matrixsum, channel_ensemble, list2matsum
except ImportError:
    from channel_IR import Matrixsum, channel_ensemble, list2matsum


def _site_label(n: int, i: int, pauli: str) -> str:
    """Build an n-qubit Pauli label with `pauli` acting on bit-index i (little-endian)."""
    chars = ["I"] * n
    chars[n - 1 - i] = pauli
    return "".join(chars)


def hypercube_T_ib_paulisum(n: int, i: int, b: int, dagger: bool = False) -> Matrixsum:
    """
    Section 5.2 operator T_{i,b} in Pauli-sum form only.

    T_{i,b} sets bit i to b in computational basis.
    On site i:
      T_{i,b} = 1/2 * (I + X + s Z + i s Y), s = (-1)^b
      T_{i,b}^dagger = 1/2 * (I + X + s Z - i s Y)
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if i < 0 or i >= n:
        raise ValueError("i must satisfy 0 <= i < n.")
    if b not in (0, 1):
        raise ValueError("b must be 0 or 1.")

    s = 1 if b == 0 else -1
    y_coeff = (-1j * s) if dagger else (1j * s)

    return list2matsum(
        [
            (_site_label(n, i, "I"), 0.5),
            (_site_label(n, i, "X"), 0.5),
            (_site_label(n, i, "Z"), 0.5 * s),
            (_site_label(n, i, "Y"), 0.5 * y_coeff),
        ]
    )


def hypercube_section52_kraus_paulisum(n: int, i: int, b: int) -> Matrixsum:
    """
    Standard Kraus form for Sec. 5.2 channel:
      K_{i,b} = T_{i,b}^dagger / sqrt(2n)
    """
    kraus = hypercube_T_ib_paulisum(n=n, i=i, b=b, dagger=True)
    kraus.mul_coeffs(1 / np.sqrt(2 * n))
    return kraus


def build_hypercube_section52_channel(n: int) -> channel_ensemble:
    """
    Build Sec. 5.2 channel as a single-channel ensemble:
      E(rho) = sum_{i=0}^{n-1} sum_{b in {0,1}} K_{i,b} rho K_{i,b}^dagger
    where K_{i,b} = T_{i,b}^dagger / sqrt(2n).
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    kraus_ops = []
    for i in range(n):
        for b in (0, 1):
            kraus_ops.append(hypercube_section52_kraus_paulisum(n, i, b))

    return channel_ensemble([kraus_ops])


def print_kraus_terms_count(n: int) -> None:
    """Static check: print Matrixsum term counts for all Kraus operators."""
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]

    print(f"n={n}, total_kraus={len(kraus_ops)}")
    idx = 0
    for i in range(n):
        for b in (0, 1):
            terms = len(kraus_ops[idx].instances)
            print(kraus_ops[idx])
            print(f"K_{{{i},{b}}}: matrixsum_terms={terms}")
            idx += 1


def _embed_sys_input(ctrl_size: int, sys_state: np.ndarray) -> np.ndarray:
    """Build |0...0>_ctrl \\otimes |psi>_sys in full Hilbert space by qubit index."""
    sys_size = int(np.log2(len(sys_state)))
    total = ctrl_size + sys_size
    full = np.zeros(2**total, dtype=complex)

    sys_qubits = list(range(ctrl_size, ctrl_size + sys_size))
    for sys_idx, amp in enumerate(sys_state):
        flat_idx = 0
        for bit_pos, qid in enumerate(sys_qubits):
            if ((sys_idx >> bit_pos) & 1) == 1:
                flat_idx |= (1 << qid)
        full[flat_idx] = amp

    return full


def _postselect_ctrl_zero(final_sv: Statevector, ctrl_size: int, sys_size: int):
    """Return normalized postselected system vector and success probability for ctrl=0."""
    data = np.asarray(final_sv.data, dtype=complex)
    out = np.zeros(2**sys_size, dtype=complex)

    sys_qubits = list(range(ctrl_size, ctrl_size + sys_size))
    for sys_idx in range(2**sys_size):
        flat_idx = 0
        for bit_pos, qid in enumerate(sys_qubits):
            if ((sys_idx >> bit_pos) & 1) == 1:
                flat_idx |= (1 << qid)
        out[sys_idx] = data[flat_idx]

    prob = float(np.vdot(out, out).real)
    if prob > 1e-14:
        out = out / np.sqrt(prob)
    return out, prob


def _trace_distance_from_statevec(psi: np.ndarray, phi: np.ndarray) -> float:
    """Trace distance between pure states |psi><psi| and |phi><phi|."""
    rho = np.outer(psi, np.conj(psi))
    sigma = np.outer(phi, np.conj(phi))
    svals = np.linalg.svd(rho - sigma, compute_uv=False)
    return 0.5 * float(np.sum(np.abs(svals)))


def _apply_channel_direct(channel_kraus: list, rho: np.ndarray) -> np.ndarray:
    """Apply rho -> sum_j K_j rho K_j^dagger with Matrixsum Kraus operators."""
    out = np.zeros_like(rho, dtype=complex)
    for kms in channel_kraus:
        K = kms.eff_op().to_matrix()
        out += K @ rho @ K.conj().T
    return out


def _trace_distance_density(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance 1/2 * ||rho - sigma||_1 for density matrices."""
    diff = rho - sigma
    svals = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * float(np.sum(np.abs(svals)))


def test_block_encoding_no_vs_matrix_order(n: int, seed: int = 7, tol: float = 1e-8) -> None:
    """
    For each Kraus Matrixsum in Sec. 5.2 channel:
    1) build block-encoding with opt='No' and opt='Matrix-order'
    2) apply both circuits to the same random system input
    3) compare postselected ctrl=0 system outputs
    4) print mobius_phase_result for Matrix-order
    """
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]

    rng = np.random.default_rng(seed)

    print(f"[test] n={n}, total_kraus={len(kraus_ops)}, seed={seed}, trace_distance_tol={tol}")
    idx = 0
    all_pass = True
    for i in range(n):
        for b in (0, 1):
            ms = kraus_ops[idx]

            be_no = BlockEncoding(ms)
            qc_no = be_no.circuit(opt='No')

            be_mo = BlockEncoding(ms)
            qc_mo = be_mo.circuit(opt='Matrix-order')

            sys_size = ms.size
            ctrl_no = qc_no.num_qubits - sys_size
            ctrl_mo = qc_mo.num_qubits - sys_size

            psi = rng.normal(size=2**sys_size) + 1j * rng.normal(size=2**sys_size)
            psi = psi / np.linalg.norm(psi)

            ini_no = Statevector(_embed_sys_input(ctrl_no, psi))
            ini_mo = Statevector(_embed_sys_input(ctrl_mo, psi))

            out_no, p_no = _postselect_ctrl_zero(ini_no.evolve(qc_no), ctrl_no, sys_size)
            out_mo, p_mo = _postselect_ctrl_zero(ini_mo.evolve(qc_mo), ctrl_mo, sys_size)

            trace_dist = _trace_distance_from_statevec(out_no, out_mo)
            passed = trace_dist <= tol
            all_pass = all_pass and passed

            print(f"K_{{{i},{b}}}: ctrl_no={ctrl_no}, ctrl_mo={ctrl_mo}, p_no={p_no:.6e}, p_mo={p_mo:.6e}, trace_distance={trace_dist:.12e}, pass={passed}")
            print(f"K_{{{i},{b}}} mobius_phase_result: {be_mo.mobius_phase_result}")

            idx += 1

    print(f"[summary] all_pass={all_pass}")


def test_kraus_to_instruction(n: int = 3, seed: int = 7) -> None:
    """
    Build channel with qiskit.Kraus -> to_instruction(), then transpile and run.
    Report accuracy against direct Kraus action and circuit metrics.
    """
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]
    kraus_mats = [kms.eff_op().to_matrix() for kms in kraus_ops]

    kraus_channel = Kraus(kraus_mats)  # type: ignore[arg-type]
    kraus_inst = kraus_channel.to_instruction()

    qc = QuantumCircuit(n, name=f"hypercube_kraus_n{n}")
    qc.append(kraus_inst, range(n))

    try:
        tqc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=3)
        transpile_mode = "basis=[cx,u3], opt=3"
    except Exception as exc:
        # Some Qiskit versions cannot synthesize Kraus instruction into cx/u3 basis.
        tqc = transpile(qc, optimization_level=3)
        transpile_mode = f"fallback-default(opt=3) due to {type(exc).__name__}"
    ops = {str(k): int(v) for k, v in tqc.count_ops().items()}

    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi = psi / np.linalg.norm(psi)
    rho0 = np.outer(psi, np.conj(psi))

    rho_target = _apply_channel_direct(kraus_ops, rho0)
    rho_target = rho_target / np.trace(rho_target)

    # Sanity reference: quantum_info channel evolution should match direct Kraus sum.
    rho_qi = DensityMatrix(rho0).evolve(kraus_channel).data

    sim = AerSimulator(method="density_matrix")
    qc_run = tqc.copy()
    qc_run.save_density_matrix(label="rho_out")  # type: ignore[attr-defined]
    result = sim.run(qc_run, initial_statevector=psi, shots=1).result().data(0)
    rho_sim = np.asarray(result["rho_out"], dtype=complex)

    td_direct_vs_qi = _trace_distance_density(rho_qi, rho_target)
    td_instruction_vs_direct = _trace_distance_density(rho_sim, rho_target)
    td_instruction_vs_qi = _trace_distance_density(rho_sim, rho_qi)

    print(f"[kraus-test] n={n}, seed={seed}")
    print(f"transpile_mode={transpile_mode}")
    print(f"instruction_qubits={qc.num_qubits}, transpiled_qubits={tqc.num_qubits}")
    print(f"depth={tqc.depth()}, size={tqc.size()}")
    print(f"gatecount={ops}")
    print(f"trace_distance(direct, qiskit_channel)={td_direct_vs_qi:.12e}")
    print(f"trace_distance(instruction, direct)={td_instruction_vs_direct:.12e}")
    print(f"trace_distance(instruction, qiskit_channel)={td_instruction_vs_qi:.12e}")


def test_stinespring_isometry(n: int = 3, seed: int = 7) -> None:
    """
    Build a decomposable circuit via Stinespring isometry and report circuit resources.
    """
    ensemble = build_hypercube_section52_channel(n)
    kraus_ops = ensemble.channels[0][1]
    kraus_mats = [kms.eff_op().to_matrix() for kms in kraus_ops]

    d_in = 2**n
    d_out = d_in
    d_env = len(kraus_mats)
    env_qubits = int(np.ceil(np.log2(d_env)))
    d_env_pad = 2**env_qubits

    # Canonical Stinespring isometry V = sum_k |k> \otimes K_k.
    V = np.vstack(kraus_mats)
    A_pad = np.zeros((d_out * d_env_pad, d_in), dtype=complex)
    A_pad[: d_out * d_env, :] = V

    # Synthesize the isometry as a decomposable circuit.
    # IMPORTANT: qargs are reversed to match Isometry's internal qubit ordering convention.
    iso = Isometry(A_pad, num_ancillas_zero=0, num_ancillas_dirty=0)
    qc = QuantumCircuit(env_qubits + n, name=f"hypercube_stine_iso_n{n}")
    qc.append(iso, list(range(env_qubits + n))[::-1])

    tqc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=0)
    ops = {str(k): int(v) for k, v in tqc.count_ops().items()}

    print(f"[stinespring-iso-test] n={n}, seed={seed}")
    print(f"env_dim={d_env}, env_qubits={env_qubits}, env_dim_padded={d_env_pad}")
    print(f"instruction_qubits={qc.num_qubits}, transpiled_qubits={tqc.num_qubits}")
    print(f"depth={tqc.depth()}, size={tqc.size()}")
    print(f"gatecount={ops}")


__all__ = [
    "hypercube_T_ib_paulisum",
    "hypercube_section52_kraus_paulisum",
    "build_hypercube_section52_channel",
    "print_kraus_terms_count",
    "test_block_encoding_no_vs_matrix_order",
    "test_kraus_to_instruction",
    "test_stinespring_isometry",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print Matrixsum term counts for all Kraus operators in Sec. 5.2 hypercube channel."
    )
    parser.add_argument("--n", type=int, default=4, help="Hypercube dimension n (default: 4)")
    parser.add_argument("--test-be", action="store_true", help="Run block-encoding consistency test (No vs Matrix-order)")
    parser.add_argument("--test-kraus", action="store_true", help="Run qiskit.Kraus to_instruction benchmark")
    parser.add_argument("--test-stine", action="store_true", help="Run Stinespring-isometry decomposable circuit test")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for tests")
    parser.add_argument("--tol", type=float, default=1e-8, help="Tolerance on trace distance for --test-be")
    args = parser.parse_args()
    if args.test_be:
        test_block_encoding_no_vs_matrix_order(args.n, seed=args.seed, tol=args.tol)
    elif args.test_kraus:
        test_kraus_to_instruction(args.n, seed=args.seed)
    elif args.test_stine:
        test_stinespring_isometry(args.n, seed=args.seed)
    else:
        print_kraus_terms_count(args.n)
