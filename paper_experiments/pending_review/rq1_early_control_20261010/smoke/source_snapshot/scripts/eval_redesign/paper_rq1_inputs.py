"""Reconstruct the 18 RQ1 appendix rows, retaining historical random dimensions.

The two random rows each expand to seeds 0..4. TFIM-3 is an extra diagnostic.
Decay-HO materializes the same finite Duhamel/Taylor operators as Pauli sums;
the legacy table used concatenated gadgets, so its gate counts are not a direct
before/after comparison with this shared channel-input protocol.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path


def case_names():
    return (["TFIM-3", "TFIM-4", "TFIM-8", "TFIM-12"]
            + [f"HRW-{n}" for n in (4, 8, 12, 20, 28)]
            + ["Toom-L2", "Toom-L3", "GC-2x4", "GC-bd3-12", "Decay-HO",
               "RepDQE-d3", "RepDQE-d5", "Color-n4q4-lin"]
            + [f"RandCh-N{n}-actual{q}q-seed{s}" for n, q in ((5, 2), (9, 3)) for s in range(5)])


def decay_kraus():
    import numpy as np
    from channel_IR import Matrixsum, PauliAtom, matsum_mul
    from build_decay_higher_order_expansion import build_decay_lindbladian
    from series_expansion import create_index_set, quadrature_points_N_weights
    lind = build_decay_lindbladian(1.0, 0.5, 4.01)
    generator = lind.effective_H()
    jumps = [op.eff_op().to_matrix() for op in lind.L_list]

    def coherent(interval):
        # Match the legacy builder's arithmetic, including matsum_mul's
        # coefficient rounding, before materializing the composed operator.
        total = generator.identity(generator.size)
        term = None
        for degree in range(1, 5):
            term = deepcopy(generator) if degree == 1 else matsum_mul(term, generator)
            term.mul_coeffs(interval / degree)
            total = total.add(term)
        return total.eff_op().to_matrix()

    matrices = [coherent(0.1)]
    indices = [[0]]
    for order in (1, 2):
        for index in create_index_set(order, 4, len(jumps)):
            points, weights = quadrature_points_N_weights(order, 4, 0.1, index[order+1:2*order+1])
            points.append(0)
            product = np.eye(2, dtype=complex)
            for segment in range(order + 1):
                interval = 0.1 - points[0] if segment == order else points[order-segment-1] - points[order-segment]
                product = coherent(interval) @ product
                if segment < order:
                    product = jumps[index[1:order+1][order-segment-1]] @ product
            matrices.append(np.sqrt(np.prod(weights)) * product)
            indices.append(index)
    from qiskit.quantum_info import Pauli
    paulis = [(label, Pauli(label).to_matrix()) for label in "IXYZ"]
    rows = []
    for matrix in matrices:
        terms = []
        for label, p in paulis:
            value = np.trace(p @ matrix) / 2
            if value != 0:
                terms.append((PauliAtom(label, phase=value / abs(value)), float(abs(value))))
        rows.append(Matrixsum(terms))
    error = max(float(np.linalg.norm((ms.eff_op().to_matrix() if ms.instances else np.zeros((2, 2))) - matrix))
                for ms, matrix in zip(rows, matrices))
    if error > 1e-12:
        raise ValueError(f"Decay Pauli reconstruction failed: {error}")
    nonzero = [row for row in rows if row.instances]
    return nonzero, {"lambda": 1.0, "v": 0.5, "scaling": 4.01, "total_time": 0.1,
                  "K": 2, "q": 4, "K1": 4, "system_qubits": 1,
                  "representation": "materialized finite Duhamel Kraus Pauli sums; legacy circuit used concatenated gadgets",
                  "logical_kraus_count": len(rows), "exact_zero_kraus_removed": len(rows)-len(nonzero), "indices": indices,
                  "dense_to_pauli_max_error": error}


def build_case(name):
    if name not in case_names():
        raise ValueError(name)
    if name.startswith("TFIM-"):
        from run_tfim4_channel_lcu_ours_nam import build_periodic_tfim_channel
        raw, model = build_periodic_tfim_channel(int(name.split("-")[1]), 0.1)
        sources = ["run_tfim4_channel_lcu_ours_nam.py"]
    elif name.startswith("HRW-"):
        from run_hypercube_n4_ours_nam import build_hypercube_channel
        n = int(name.split("-")[1])
        raw, model = build_hypercube_channel(n), {"system_qubits": n, "definition": "HRW section 5.2"}
        sources = ["run_hypercube_n4_ours_nam.py"]
    elif name.startswith("Toom-"):
        from build_toom_2d_channel_lcu import build_toom_channel
        raw, model = build_toom_channel(int(name[-1]), 0.1, 0.1)
        sources = ["build_toom_2d_channel_lcu.py"]
    elif name.startswith("GC-"):
        from build_graph_cluster_state_channel_lcu import grid_2d_cluster_graph, random_bounded_degree_graph, build_channel_for_spec
        spec = grid_2d_cluster_graph(2, 4) if name == "GC-2x4" else random_bounded_degree_graph(12, 3, 7)
        raw, model = build_channel_for_spec(spec, 0.1, 0.1)
        sources = ["build_graph_cluster_state_channel_lcu.py"]
    elif name.startswith("RepDQE-"):
        from build_repetition_code_dqe_channel import DqeSpec, build_repetition_dqe_channel, pauli_dict_to_matrixsum
        rows, model = build_repetition_dqe_channel(DqeSpec(d=int(name[-1]), epsilon=0.25))
        raw = [pauli_dict_to_matrixsum(row) for row in rows]
        sources = ["build_repetition_code_dqe_channel.py"]
    elif name == "Color-n4q4-lin":
        from run_random_coloring_channel_lcu_ours_nam import build_random_coloring_channel
        raw, model = build_random_coloring_channel(4, "linear", q=4, tol=1e-12,
                      max_total_pauli_terms=250000, max_terms_per_kraus=50000,
                      color_mixing_preprocess=False)
        sources = ["run_random_coloring_channel_lcu_ours_nam.py"]
    elif name == "Decay-HO":
        raw, model = decay_kraus()
        sources = ["build_decay_higher_order_expansion.py", "build_periodic_tfim_higher_order_expansion.py"]
    else:
        from build_random_channel_lcu_benchmark import RandomChannelSpec, RandomChannelCase, build_random_kraus_ops
        _, count, width, seed = name.split("-")
        n, q, seed = int(count[1:]), int(width[6:-1]), int(seed[4:])
        spec = RandomChannelSpec.from_qubits(kraus_count=n, num_qubits=q,
                   first_kraus_terms=10 if n == 5 else 20, other_kraus_terms=2 if n == 5 else 4)
        raw, model = build_random_kraus_ops(RandomChannelCase(spec, seed))
        model["historical_input_policy"] = "RQ1 original dim4/dim8 preserved as explicit 2q/3q; D07 4q correction is a separate dataset"
        sources = ["build_random_channel_lcu_benchmark.py"]
    root = Path(__file__).resolve().parents[2]
    model.update({"suite_case": name, "system_qubits": raw[0].size,
                  "builder_sources": {s: sha256((root / "scripts" / s).read_bytes()).hexdigest() for s in sources},
                  "suite_builder_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})
    return deepcopy(raw), model
