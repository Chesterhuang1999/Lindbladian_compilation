"""Multi-candidate variant of `BlockEncoding`.

Differs from the legacy `block_encoding.BlockEncoding` only in
`find_optimal_order_matrices`, which evaluates two candidate basis sizes
(`k = w` and `k = w - 1`) and selects the one with smaller weighted control
cost `C = Σ w(ctrl) · w_s(g_l)` (Möbius output). All other behaviour
(circuit synthesis, resource counting, etc.) is inherited unchanged.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.quantum_info import PauliList

from block_encoding import (
    BlockEncoding as _LegacyBlockEncoding,
    sort_symplectic_matrix,
    greedy_generator_selection,
    assign_subspace_modes,
    assign_additional_modes,
    build_vec_phase_lookup,
    extract_basis_modes_with_phases,
    mobius_invert_modes_with_phases,
)


def _ctrl_state_bits(gate: ControlledGate) -> str:
    """Return control-state bits in Qiskit's control-string convention."""
    ctrl_count = int(gate.num_ctrl_qubits)
    ctrl_state = gate.ctrl_state
    if ctrl_state is None:
        ctrl_state = (1 << ctrl_count) - 1
    return format(int(ctrl_state), f"0{ctrl_count}b")


def _instruction_controls_targets(instruction):
    op = instruction.operation
    qargs = list(instruction.qubits)
    if isinstance(op, ControlledGate):
        ctrl_count = int(op.num_ctrl_qubits)
        return qargs[:ctrl_count], qargs[ctrl_count:]
    return [], qargs


def _is_controlled_x_gate(op) -> bool:
    return isinstance(op, ControlledGate) and getattr(op.base_gate, "name", None) == "x"


def _clean_pauli_label(label: object, qubit_count: int) -> str | None:
    if label is None:
        return None
    cleaned = str(label).replace(" ", "")
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if len(cleaned) != qubit_count:
        return None
    cleaned = cleaned.upper()
    if any(char not in {"I", "X", "Y", "Z"} for char in cleaned):
        return None
    return cleaned


def _pauli_label_from_operation(operation, qubit_count: int) -> str | None:
    label_method = getattr(operation, "to_label", None)
    if callable(label_method):
        label = _clean_pauli_label(label_method(), qubit_count)
        if label is not None:
            return label

    for attr_name in ("label", "name"):
        label = _clean_pauli_label(getattr(operation, attr_name, None), qubit_count)
        if label is not None:
            return label

    for param in getattr(operation, "params", ()):
        label = _clean_pauli_label(param, qubit_count)
        if label is not None:
            return label
        label_method = getattr(param, "to_label", None)
        if callable(label_method):
            label = _clean_pauli_label(label_method(), qubit_count)
            if label is not None:
                return label

    return None


def _pauli_ops_from_label(label: str) -> list[tuple[str, int]]:
    return [(char, index) for index, char in enumerate(label) if char != "I"]


def _single_qubit_pauli_data_from_definition(
    operation,
    seen: set[int] | None = None,
) -> dict[str, object] | None:
    if seen is None:
        seen = set()
    operation_id = id(operation)
    if operation_id in seen:
        return None
    seen.add(operation_id)

    definition = getattr(operation, "definition", None)
    if definition is None:
        return None

    pauli_by_index = ["I"] * int(operation.num_qubits)
    phase_angle = getattr(definition, "global_phase", 0)

    def apply_pauli(target_index: int, pauli: str) -> bool:
        nonlocal phase_angle
        if target_index < 0 or target_index >= len(pauli_by_index):
            return False
        if pauli == "I":
            return True
        current = pauli_by_index[target_index]
        if current == "I":
            pauli_by_index[target_index] = pauli
            return True
        if current == pauli:
            pauli_by_index[target_index] = "I"
            return True

        product_table = {
            ("X", "Y"): ("Z", np.pi / 2),
            ("Y", "X"): ("Z", -np.pi / 2),
            ("Y", "Z"): ("X", np.pi / 2),
            ("Z", "Y"): ("X", -np.pi / 2),
            ("Z", "X"): ("Y", np.pi / 2),
            ("X", "Z"): ("Y", -np.pi / 2),
        }
        result = product_table.get((pauli, current))
        if result is None:
            return False
        pauli_by_index[target_index], phase_delta = result
        phase_angle += phase_delta
        return True

    for instruction in definition.data:
        if instruction.clbits:
            return None
        op = instruction.operation
        qargs = list(instruction.qubits)
        label = _pauli_label_from_operation(op, len(qargs))
        if label is not None:
            for offset, char in enumerate(label):
                target_index = definition.find_bit(qargs[offset]).index
                if not apply_pauli(target_index, char):
                    return None
            continue

        nested_pauli_data = _single_qubit_pauli_data(op, set(seen))
        if nested_pauli_data is not None:
            phase_angle += nested_pauli_data["phase"]
            for char, offset in nested_pauli_data["ops"]:
                if offset >= len(qargs):
                    return None
                target_index = definition.find_bit(qargs[offset]).index
                if not apply_pauli(target_index, char):
                    return None
            continue

        name = getattr(op, "name", None)
        if len(qargs) != 1 or name not in {"x", "y", "z"}:
            return None
        target_index = definition.find_bit(qargs[0]).index
        if not apply_pauli(target_index, str(name).upper()):
            return None

    return {
        "ops": [(char, index) for index, char in enumerate(pauli_by_index) if char != "I"],
        "phase": phase_angle,
    }


def _single_qubit_pauli_data(
    operation,
    seen: set[int] | None = None,
) -> dict[str, object] | None:
    label = _pauli_label_from_operation(operation, int(operation.num_qubits))
    if label is not None:
        return {
            "ops": _pauli_ops_from_label(label),
            "phase": 0,
        }
    return _single_qubit_pauli_data_from_definition(operation, seen)


def _controlled_gate_single_qubit_pauli_data(gate: ControlledGate) -> dict[str, object] | None:
    target_count = int(gate.num_qubits) - int(gate.num_ctrl_qubits)
    base_operation = getattr(gate, "base_gate", None)
    if base_operation is None:
        return None
    pauli_data = _single_qubit_pauli_data(base_operation)
    if pauli_data is None:
        return None
    pauli_ops = pauli_data["ops"]
    if any(target_index >= target_count for _pauli, target_index in pauli_ops):
        return None
    return pauli_data


def _is_ancilla_target_controlled_gate(instruction, ancilla_qubits: set) -> bool:
    op = instruction.operation
    if not isinstance(op, ControlledGate):
        return False
    _controls, targets = _instruction_controls_targets(instruction)
    return len(targets) == 1 and targets[0] in ancilla_qubits


def _is_optimizable_ancilla_target_controlled_gate(instruction, ancilla_qubits: set) -> bool:
    op = instruction.operation
    if instruction.clbits:
        return False
    if not _is_ancilla_target_controlled_gate(instruction, ancilla_qubits):
        return False
    return _is_controlled_x_gate(op) and int(op.num_ctrl_qubits) <= 2


def _can_move_controlled_gate_right(moving_instruction, other_instruction) -> bool:
    moving_controls, moving_targets = _instruction_controls_targets(moving_instruction)
    other_controls, other_targets = _instruction_controls_targets(other_instruction)

    moving_control_set = set(moving_controls)
    moving_target_set = set(moving_targets)
    other_control_set = set(other_controls)
    other_target_set = set(other_targets)

    if moving_target_set & other_control_set:
        return False
    if moving_control_set & other_target_set:
        return False
    if moving_target_set & other_target_set:
        return _is_controlled_x_gate(moving_instruction.operation) and _is_controlled_x_gate(
            other_instruction.operation
        )
    return True


def _copy_circuit_with_instructions(qc: QuantumCircuit, instructions: list) -> QuantumCircuit:
    copied = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    copied.global_phase = qc.global_phase
    copied.metadata = None if qc.metadata is None else dict(qc.metadata)
    for instruction in instructions:
        copied.append(
            instruction.operation,
            list(instruction.qubits),
            list(instruction.clbits),
        )
    return copied


def _append_instruction(qc: QuantumCircuit, instruction) -> None:
    qc.append(
        instruction.operation,
        list(instruction.qubits),
        list(instruction.clbits),
    )


def _qubit_label(qc: QuantumCircuit, qubit) -> str:
    location = qc.find_bit(qubit)
    registers = getattr(location, "registers", ())
    if registers:
        reg, index = registers[0]
        return f"{reg.name}[{index}]"
    return f"q[{location.index}]"


def _control_literals(qc: QuantumCircuit, instruction) -> list[dict[str, object]]:
    controls, _targets = _instruction_controls_targets(instruction)
    ctrl_bits = _ctrl_state_bits(instruction.operation)[::-1]
    return [
        {
            "qubit": _qubit_label(qc, qubit),
            "positive": bit == "1",
        }
        for qubit, bit in zip(controls, ctrl_bits)
    ]


def _format_product_expression(literals: list[dict[str, object]]) -> str:
    if not literals:
        return "1"
    terms = [
        str(literal["qubit"]) if literal["positive"] else f"~{literal['qubit']}"
        for literal in literals
    ]
    if len(terms) == 1:
        return terms[0]
    return "(" + " & ".join(terms) + ")"


def _anf_terms_from_product_literals(
    product_literals: list[list[dict[str, object]]],
) -> set[frozenset[str]]:
    def multiply_by_positive(poly: set[frozenset[str]], variable: str) -> set[frozenset[str]]:
        return {frozenset(set(monomial) | {variable}) for monomial in poly}

    def multiply_by_negative(poly: set[frozenset[str]], variable: str) -> set[frozenset[str]]:
        positive_part = multiply_by_positive(poly, variable)
        return poly ^ positive_part

    def product_to_anf(literals: list[dict[str, object]]) -> set[frozenset[str]]:
        poly = {frozenset()}
        for literal in literals:
            variable = str(literal["qubit"])
            if literal["positive"]:
                poly = multiply_by_positive(poly, variable)
            else:
                poly = multiply_by_negative(poly, variable)
        return poly

    anf_terms: set[frozenset[str]] = set()
    for literals in product_literals:
        anf_terms ^= product_to_anf(literals)
    return anf_terms


def _monomial_sort_key(
    monomial: frozenset[str],
    control_order: dict[str, int],
) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
    ordered = sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
    return (len(ordered), tuple(control_order.get(label, len(control_order)) for label in ordered), tuple(ordered))


def _monomial_expr(monomial: frozenset[str], control_order: dict[str, int]):
    if len(monomial) == 0:
        return {"kind": "const", "value": 1}
    ordered = sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
    if len(ordered) == 1:
        return {"kind": "var", "name": ordered[0]}
    return {"kind": "and", "terms": [{"kind": "var", "name": label} for label in ordered]}


def _xor_expr(terms: list[dict]) -> dict:
    flat_terms = []
    for term in terms:
        if term["kind"] == "const" and term["value"] == 0:
            continue
        if term["kind"] == "xor":
            flat_terms.extend(term["terms"])
        else:
            flat_terms.append(term)
    if not flat_terms:
        return {"kind": "const", "value": 0}
    if len(flat_terms) == 1:
        return flat_terms[0]
    return {"kind": "xor", "terms": flat_terms}


def _and_expr(terms: list[dict]) -> dict:
    flat_terms = []
    for term in terms:
        if term["kind"] == "const":
            if term["value"] == 0:
                return {"kind": "const", "value": 0}
            continue
        if term["kind"] == "and":
            flat_terms.extend(term["terms"])
        else:
            flat_terms.append(term)
    if not flat_terms:
        return {"kind": "const", "value": 1}
    if len(flat_terms) == 1:
        return flat_terms[0]
    return {"kind": "and", "terms": flat_terms}


def _expr_to_string(expr: dict) -> str:
    kind = expr["kind"]
    if kind == "const":
        return str(expr["value"])
    if kind == "var":
        return str(expr["name"])
    if kind == "xor":
        return " ^ ".join(_expr_to_string(term) for term in expr["terms"])
    if kind == "and":
        pieces = []
        for term in expr["terms"]:
            piece = _expr_to_string(term)
            if term["kind"] == "xor":
                piece = f"({piece})"
            pieces.append(piece)
        return " & ".join(pieces)
    raise ValueError(f"Unsupported expression kind: {kind}")


def _expr_cost(expr: dict) -> tuple[int, int]:
    kind = expr["kind"]
    if kind == "const":
        return (0 if expr["value"] == 0 else 1, 1)
    if kind == "var":
        return (1, len(str(expr["name"])))
    if kind == "xor":
        child_costs = [_expr_cost(term) for term in expr["terms"]]
        return (sum(cost for cost, _ in child_costs), len(_expr_to_string(expr)))
    if kind == "and":
        child_costs = [_expr_cost(term) for term in expr["terms"]]
        non_var_count = sum(1 for term in expr["terms"] if term["kind"] != "var")
        return (sum(cost for cost, _ in child_costs) + max(len(expr["terms"]) - 1, 1) + 2 * non_var_count, len(_expr_to_string(expr)))
    raise ValueError(f"Unsupported expression kind: {kind}")


def _candidate_common_factors(terms: set[frozenset[str]]) -> list[frozenset[str]]:
    from itertools import combinations

    factors = set()
    term_list = list(terms)
    for left, right in combinations(term_list, 2):
        common = sorted(set(left) & set(right))
        for size in range(1, len(common) + 1):
            for subset in combinations(common, size):
                factors.add(frozenset(subset))
    return sorted(factors, key=lambda factor: (-len(factor), tuple(sorted(factor))))


def _factor_anf_terms_to_expr(
    anf_terms: set[frozenset[str]],
    control_order: dict[str, int],
    memo: dict[frozenset[frozenset[str]], dict] | None = None,
) -> dict:
    if memo is None:
        memo = {}
    key = frozenset(anf_terms)
    if key in memo:
        return memo[key]

    if not anf_terms:
        result = {"kind": "const", "value": 0}
        memo[key] = result
        return result

    base_expr = _xor_expr(
        [
            _monomial_expr(monomial, control_order)
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ]
    )
    best_expr = base_expr
    best_cost = _expr_cost(base_expr)

    for common in _candidate_common_factors(anf_terms):
        group = {term for term in anf_terms if common <= term}
        if len(group) < 2:
            continue
        residual_terms = {frozenset(set(term) - set(common)) for term in group}
        remaining_terms = anf_terms - group
        factored_group = _and_expr(
            [
                _factor_anf_terms_to_expr(residual_terms, control_order, memo),
                _monomial_expr(common, control_order),
            ]
        )
        candidate_expr = _xor_expr(
            [
                _factor_anf_terms_to_expr(remaining_terms, control_order, memo),
                factored_group,
            ]
        )
        candidate_cost = _expr_cost(candidate_expr)
        if candidate_cost < best_cost:
            best_expr = candidate_expr
            best_cost = candidate_cost

    memo[key] = best_expr
    return best_expr


def _gate_sequence_for_boolean_expr(expr: dict, target: str) -> list[str]:
    steps = []
    temp_index = 0

    def new_temp() -> str:
        nonlocal temp_index
        temp = f"tmp{temp_index}"
        temp_index += 1
        return temp

    def inverse_step(step: str) -> str:
        if step.startswith("ALLOC(") and step.endswith("=0)"):
            return f"FREE({step[len('ALLOC('):-len('=0)')]})"
        if step.startswith("FREE(") and step.endswith(")"):
            return f"ALLOC({step[len('FREE('):-1]}=0)"
        return step

    def inverse_steps(step_list: list[str]) -> list[str]:
        return [inverse_step(step) for step in reversed(step_list)]

    def linear_xor_terms(subexpr: dict) -> tuple[list[str], int] | None:
        if subexpr["kind"] == "var":
            return [str(subexpr["name"])], 0
        if subexpr["kind"] == "const":
            return [], int(subexpr["value"]) & 1
        if subexpr["kind"] != "xor":
            return None

        variables = []
        const_value = 0
        for term in subexpr["terms"]:
            term_linear = linear_xor_terms(term)
            if term_linear is None:
                return None
            term_variables, term_const = term_linear
            const_value ^= term_const
            for variable in term_variables:
                if variable in variables:
                    variables.remove(variable)
                else:
                    variables.append(variable)
        return variables, const_value

    def synthesize_linear_xor_control(subexpr: dict) -> tuple[str | None, list[str], list[str]] | None:
        linear_terms = linear_xor_terms(subexpr)
        if linear_terms is None:
            return None
        variables, const_value = linear_terms
        if len(variables) == 0:
            return (None, [], []) if const_value == 0 else ("1", [], [])

        accumulator = variables[-1]
        compute_steps = [f"CNOT({variable}, {accumulator})" for variable in variables[:-1]]
        if const_value:
            compute_steps.append(f"X({accumulator})")
        cleanup_steps = list(reversed(compute_steps))
        return accumulator, compute_steps, cleanup_steps

    def synthesize_toggle(subexpr: dict, out: str) -> list[str]:
        kind = subexpr["kind"]
        if kind == "const":
            return [] if subexpr["value"] == 0 else [f"X({out})"]
        if kind == "var":
            return [f"CNOT({subexpr['name']}, {out})"]
        if kind == "xor":
            result = []
            for term in subexpr["terms"]:
                result.extend(synthesize_toggle(term, out))
            return result
        if kind == "and":
            return synthesize_and(subexpr["terms"], out)
        raise ValueError(f"Unsupported expression kind: {kind}")

    def synthesize_and(factors: list[dict], out: str) -> list[str]:
        local_steps = []
        cleanup_steps = []
        controls = []
        for factor in factors:
            if factor["kind"] == "var":
                controls.append(str(factor["name"]))
            elif factor["kind"] == "const" and factor["value"] == 1:
                continue
            elif factor["kind"] == "const" and factor["value"] == 0:
                return []
            else:
                linear_control = synthesize_linear_xor_control(factor)
                if linear_control is not None:
                    control, compute_steps, cleanup = linear_control
                    if control is None:
                        return []
                    if control != "1":
                        local_steps.extend(compute_steps)
                        controls.append(control)
                        cleanup_steps.extend(cleanup)
                    continue

                temp = new_temp()
                compute_steps = synthesize_toggle(factor, temp)
                local_steps.append(f"ALLOC({temp}=0)")
                local_steps.extend(compute_steps)
                controls.append(temp)
                cleanup_steps.extend(inverse_steps(compute_steps))
                cleanup_steps.append(f"FREE({temp})")

        if len(controls) == 0:
            local_steps.append(f"X({out})")
        elif len(controls) == 1:
            local_steps.append(f"CNOT({controls[0]}, {out})")
        elif len(controls) == 2:
            local_steps.append(f"CCU({controls[0]}, {controls[1]}, {out}; U=X)")
        else:
            ladder = []
            ladder_cleanup = []
            prev = controls[0]
            for control in controls[1:-1]:
                temp = new_temp()
                local_steps.append(f"ALLOC({temp}=0)")
                local_steps.append(f"CCU({prev}, {control}, {temp}; U=X)")
                ladder.append(temp)
                ladder_cleanup.append(f"CCU({prev}, {control}, {temp}; U=X)")
                prev = temp
            local_steps.append(f"CCU({prev}, {controls[-1]}, {out}; U=X)")
            for temp, undo_step in zip(reversed(ladder), reversed(ladder_cleanup)):
                local_steps.append(undo_step)
                local_steps.append(f"FREE({temp})")

        local_steps.extend(cleanup_steps)
        return local_steps

    steps.extend(synthesize_toggle(expr, target))
    return steps


def _append_boolean_expr_toggle(
    qc: QuantumCircuit,
    expr: dict,
    target,
    label_to_qubit: dict[str, object],
) -> None:
    def get_qubit(label: str):
        if label not in label_to_qubit:
            raise ValueError(f"Unknown qubit label in boolean expression: {label}")
        return label_to_qubit[label]

    def linear_xor_terms(subexpr: dict) -> tuple[list[str], int] | None:
        if subexpr["kind"] == "var":
            return [str(subexpr["name"])], 0
        if subexpr["kind"] == "const":
            return [], int(subexpr["value"]) & 1
        if subexpr["kind"] != "xor":
            return None

        variables = []
        const_value = 0
        for term in subexpr["terms"]:
            term_linear = linear_xor_terms(term)
            if term_linear is None:
                return None
            term_variables, term_const = term_linear
            const_value ^= term_const
            for variable in term_variables:
                if variable in variables:
                    variables.remove(variable)
                else:
                    variables.append(variable)
        return variables, const_value

    def append_inverse_operations(operations: list[tuple[str, str, str | None]]) -> None:
        for name, left, right in reversed(operations):
            if name == "cx":
                qc.cx(get_qubit(left), get_qubit(str(right)))
            elif name == "x":
                qc.x(get_qubit(left))
            else:
                raise ValueError(f"Unsupported inverse operation: {name}")

    def synthesize_linear_xor_control(
        subexpr: dict,
        avoid_variables: set[str],
    ) -> tuple[str | None, list[tuple[str, str, str | None]]] | None:
        linear_terms = linear_xor_terms(subexpr)
        if linear_terms is None:
            return None
        variables, const_value = linear_terms
        if len(variables) == 0:
            return (None, []) if const_value == 0 else ("1", [])

        accumulator = next(
            (variable for variable in reversed(variables) if variable not in avoid_variables),
            None,
        )
        if accumulator is None:
            raise ValueError("Linear-XOR control overlaps all other controls; a clean temp would be required.")
        operations = []
        for variable in variables:
            if variable == accumulator:
                continue
            qc.cx(get_qubit(variable), get_qubit(accumulator))
            operations.append(("cx", variable, accumulator))
        if const_value:
            qc.x(get_qubit(accumulator))
            operations.append(("x", accumulator, None))
        return accumulator, operations

    def synthesize_toggle(subexpr: dict, out) -> None:
        kind = subexpr["kind"]
        if kind == "const":
            if subexpr["value"] == 1:
                qc.x(out)
            return
        if kind == "var":
            qc.cx(get_qubit(str(subexpr["name"])), out)
            return
        if kind == "xor":
            for term in subexpr["terms"]:
                synthesize_toggle(term, out)
            return
        if kind == "and":
            synthesize_and(subexpr["terms"], out)
            return
        raise ValueError(f"Unsupported expression kind: {kind}")

    def synthesize_and(factors: list[dict], out) -> None:
        linear_variables_by_factor = []
        for factor in factors:
            linear_terms = linear_xor_terms(factor)
            if linear_terms is None:
                linear_variables_by_factor.append(set())
            else:
                variables, _const_value = linear_terms
                linear_variables_by_factor.append(set(variables))

        controls = []
        cleanup_operations = []
        for factor_index, factor in enumerate(factors):
            if factor["kind"] == "var":
                controls.append(get_qubit(str(factor["name"])))
            elif factor["kind"] == "const" and factor["value"] == 1:
                continue
            elif factor["kind"] == "const" and factor["value"] == 0:
                return
            else:
                other_variables = set()
                for other_index, variables in enumerate(linear_variables_by_factor):
                    if other_index != factor_index:
                        other_variables.update(variables)
                linear_control = synthesize_linear_xor_control(factor, other_variables)
                if linear_control is None:
                    raise ValueError("Only linear-XOR factors can be used as synthesized controls.")
                control, operations = linear_control
                if control is None:
                    return
                if control != "1":
                    controls.append(get_qubit(control))
                    cleanup_operations.extend(operations)

        if len(controls) == 0:
            qc.x(out)
        elif len(controls) == 1:
            qc.cx(controls[0], out)
        elif len(controls) == 2:
            qc.ccx(controls[0], controls[1], out)
        else:
            raise ValueError("Synthesizing an AND with more than two controls requires extra clean ancillas.")

        append_inverse_operations(cleanup_operations)

    synthesize_toggle(expr, target)


def _simplify_xor_product_data(
    control_qubits: list[str],
    product_literals: list[list[dict[str, object]]],
    target: str,
) -> dict[str, object]:
    control_order = {label: idx for idx, label in enumerate(control_qubits)}
    anf_terms = _anf_terms_from_product_literals(product_literals)
    expr = _factor_anf_terms_to_expr(anf_terms, control_order)
    return {
        "anf_terms": [
            sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ],
        "expression_tree": expr,
        "simplified_expression": _expr_to_string(expr),
        "gate_sequence": _gate_sequence_for_boolean_expr(expr, target),
    }


def encode_control_block_boolean_expressions(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> list[dict[str, object]]:
    """
    Analyze ancilla-target controlled blocks and encode their boolean forms.

    The analysis first permutes each contiguous ancilla-target controlled block
    in-memory so gates with the same target are adjacent. For each target group
    with at least two gates, it encodes the target's toggling condition as the
    XOR of each gate's control-product expression and records a simplified
    boolean expression. The input circuit is not modified.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        return []

    instructions = list(qc.data)
    blocks = []
    current_block = []
    for index, instruction in enumerate(instructions):
        if _is_ancilla_target_controlled_gate(instruction, ancilla_set):
            current_block.append((index, instruction))
        elif current_block:
            blocks.append(current_block)
            current_block = []
    if current_block:
        blocks.append(current_block)

    encoded_blocks = []
    for block_id, block in enumerate(blocks):
        target_order = []
        target_groups = {}
        for instruction_index, instruction in block:
            _controls, targets = _instruction_controls_targets(instruction)
            target = targets[0]
            if target not in target_groups:
                target_order.append(target)
                target_groups[target] = []
            target_groups[target].append((instruction_index, instruction))

        permuted_indices = [
            instruction_index
            for target in target_order
            for instruction_index, _instruction in target_groups[target]
        ]

        for target in target_order:
            group = target_groups[target]
            if len(group) < 2:
                continue

            product_literals = []
            products = []
            control_qubits = []
            for instruction_index, instruction in group:
                literals = _control_literals(qc, instruction)
                product_literals.append(literals)
                products.append(
                    {
                        "gate_index": instruction_index,
                        "expression": _format_product_expression(literals),
                        "literals": literals,
                    }
                )
                for literal in literals:
                    qubit = str(literal["qubit"])
                    if qubit not in control_qubits:
                        control_qubits.append(qubit)

            xor_expression = " ^ ".join(str(product["expression"]) for product in products)
            target_label = _qubit_label(qc, target)
            simplified_data = _simplify_xor_product_data(
                control_qubits,
                product_literals,
                target_label,
            )
            encoded_blocks.append(
                {
                    "block_id": block_id,
                    "block_gate_indices": [instruction_index for instruction_index, _ in block],
                    "permuted_gate_indices": permuted_indices,
                    "target": target_label,
                    "control_qubits": control_qubits,
                    "products": products,
                    "anf_terms": simplified_data["anf_terms"],
                    "xor_expression": xor_expression,
                    "simplified_expression": simplified_data["simplified_expression"],
                    "gate_sequence": simplified_data["gate_sequence"],
                }
            )

    return encoded_blocks


def swap_phase_controlled_ancilla_targets(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> QuantumCircuit:
    """
    Move eligible ancilla-target controlled gates next to the next such gate.

    For a controlled gate whose single target is an ancilla, find the next
    controlled gate with a single ancilla target. If the two target ancillas are
    not used as controls by intervening gates, and the first gate can commute
    through the intervening instructions, move the first gate immediately before
    the second. This is the first swap phase before later cancellation passes.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        return qc

    instructions = list(qc.data)
    i = 0
    while i < len(instructions):
        first = instructions[i]
        if not _is_ancilla_target_controlled_gate(first, ancilla_set):
            i += 1
            continue

        j = i + 1
        while j < len(instructions) and not _is_ancilla_target_controlled_gate(
            instructions[j],
            ancilla_set,
        ):
            j += 1
        if j >= len(instructions):
            break
        if j == i + 1:
            i += 1
            continue

        _first_controls, first_targets = _instruction_controls_targets(first)
        _second_controls, second_targets = _instruction_controls_targets(instructions[j])
        protected_ancillas = set(first_targets) | set(second_targets)

        can_move = True
        for between in instructions[i + 1:j]:
            between_controls, _between_targets = _instruction_controls_targets(between)
            if protected_ancillas & set(between_controls):
                can_move = False
                break
            if not _can_move_controlled_gate_right(first, between):
                can_move = False
                break

        if can_move:
            moved = instructions.pop(i)
            instructions.insert(j - 1, moved)
            i = max(i - 1, 0)
        else:
            i += 1

    swapped = _copy_circuit_with_instructions(qc, instructions)
    swap_phase_controlled_ancilla_targets.last_boolean_expressions = (
        encode_control_block_boolean_expressions(swapped, ancilla_set)
    )
    return swapped


def _permute_control_block_by_target_safely(block: list[tuple[int, object]]) -> list[tuple[int, object]]:
    target_order = []
    target_rank = {}
    for _instruction_index, instruction in block:
        _controls, targets = _instruction_controls_targets(instruction)
        target = targets[0]
        if target not in target_rank:
            target_rank[target] = len(target_order)
            target_order.append(target)

    permuted = list(block)
    changed = True
    while changed:
        changed = False
        for index in range(len(permuted) - 1):
            _left_index, left = permuted[index]
            _right_index, right = permuted[index + 1]
            _left_controls, left_targets = _instruction_controls_targets(left)
            _right_controls, right_targets = _instruction_controls_targets(right)
            if target_rank[left_targets[0]] <= target_rank[right_targets[0]]:
                continue
            if _can_move_controlled_gate_right(left, right):
                permuted[index], permuted[index + 1] = permuted[index + 1], permuted[index]
                changed = True
    return permuted


def _append_simplified_control_group(
    output: QuantumCircuit,
    source: QuantumCircuit,
    group: list[tuple[int, object]],
    label_to_qubit: dict[str, object],
) -> dict[str, object]:
    _controls, targets = _instruction_controls_targets(group[0][1])
    target = targets[0]
    target_label = _qubit_label(source, target)
    product_literals = []
    products = []
    control_qubits = []
    for instruction_index, instruction in group:
        literals = _control_literals(source, instruction)
        product_literals.append(literals)
        products.append(
            {
                "gate_index": instruction_index,
                "expression": _format_product_expression(literals),
                "literals": literals,
            }
        )
        for literal in literals:
            qubit = str(literal["qubit"])
            if qubit not in control_qubits:
                control_qubits.append(qubit)

    control_order = {label: idx for idx, label in enumerate(control_qubits)}
    anf_terms = _anf_terms_from_product_literals(product_literals)
    expr = _factor_anf_terms_to_expr(anf_terms, control_order)
    _append_boolean_expr_toggle(output, expr, target, label_to_qubit)

    return {
        "target": target_label,
        "control_qubits": control_qubits,
        "products": products,
        "anf_terms": [
            sorted(monomial, key=lambda label: (control_order.get(label, len(control_order)), label))
            for monomial in sorted(
                anf_terms,
                key=lambda monomial: _monomial_sort_key(monomial, control_order),
            )
        ],
        "xor_expression": " ^ ".join(str(product["expression"]) for product in products),
        "simplified_expression": _expr_to_string(expr),
    }


def optimize_expanded_circuit(
    qc: QuantumCircuit,
    ancilla_qubits: list | tuple | set,
) -> QuantumCircuit:
    """
    Optimize expanded multi-control ladders using boolean simplification.

    The pass first runs the swap phase, then scans maximal consecutive blocks of
    ancilla-target controlled-X gates with at most two controls. Within each
    block it safely permutes gates toward target-grouped order, derives the ANF
    for each repeated target group, simplifies it, and appends the resulting
    CNOT/Toffoli implementation directly to the returned circuit.
    """
    ancilla_set = set(ancilla_qubits)
    if not ancilla_set:
        optimize_expanded_circuit.last_boolean_expressions = []
        return qc

    swapped = swap_phase_controlled_ancilla_targets(qc, ancilla_set)
    optimized = QuantumCircuit(*swapped.qregs, *swapped.cregs, name=swapped.name)
    optimized.global_phase = swapped.global_phase
    optimized.metadata = None if swapped.metadata is None else dict(swapped.metadata)

    label_to_qubit = {_qubit_label(swapped, qubit): qubit for qubit in swapped.qubits}
    instructions = list(swapped.data)
    encoded_blocks = []
    block_id = 0
    index = 0
    while index < len(instructions):
        instruction = instructions[index]
        if not _is_optimizable_ancilla_target_controlled_gate(instruction, ancilla_set):
            _append_instruction(optimized, instruction)
            index += 1
            continue

        block = []
        while index < len(instructions) and _is_optimizable_ancilla_target_controlled_gate(
            instructions[index],
            ancilla_set,
        ):
            block.append((index, instructions[index]))
            index += 1

        permuted_block = _permute_control_block_by_target_safely(block)
        permuted_indices = [instruction_index for instruction_index, _instruction in permuted_block]
        group_start = 0
        while group_start < len(permuted_block):
            _group_index, first_instruction = permuted_block[group_start]
            _first_controls, first_targets = _instruction_controls_targets(first_instruction)
            target = first_targets[0]
            group_end = group_start + 1
            while group_end < len(permuted_block):
                _next_controls, next_targets = _instruction_controls_targets(permuted_block[group_end][1])
                if next_targets[0] != target:
                    break
                group_end += 1

            group = permuted_block[group_start:group_end]
            if len(group) == 1:
                _append_instruction(optimized, group[0][1])
            else:
                entry = _append_simplified_control_group(optimized, swapped, group, label_to_qubit)
                entry.update(
                    {
                        "block_id": block_id,
                        "block_gate_indices": [instruction_index for instruction_index, _ in block],
                        "permuted_gate_indices": permuted_indices,
                    }
                )
                encoded_blocks.append(entry)
            group_start = group_end

        block_id += 1

    optimize_expanded_circuit.last_boolean_expressions = encoded_blocks
    optimize_expanded_circuit.swapped_circuit = swapped
    return optimized


def _append_full_control_pauli_expansion(
    expanded: QuantumCircuit,
    controls: list,
    targets: list,
    work: list,
    pauli_ops: list[tuple[str, int]],
    phase_angle=0,
) -> None:
    expanded.ccx(controls[0], controls[1], work[0])
    for ctrl_index in range(2, len(controls)):
        expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])

    full_control = work[len(controls) - 2]
    if phase_angle != 0:
        expanded.p(phase_angle, full_control)
    for pauli, target_index in pauli_ops:
        target = targets[target_index]
        if pauli == "X":
            expanded.cx(full_control, target)
        elif pauli == "Y":
            expanded.cy(full_control, target)
        elif pauli == "Z":
            expanded.cz(full_control, target)
        else:
            raise ValueError(f"Unsupported Pauli component: {pauli}")

    for ctrl_index in range(len(controls) - 1, 1, -1):
        expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])
    expanded.ccx(controls[0], controls[1], work[0])


def expand_large_controlled_gates(
    qc: QuantumCircuit,
    ancilla_name: str = "mc_anc",
    if_optimize = False,
) -> QuantumCircuit:
    """
    Expand every controlled gate with N >= 3 controls.

    A gate C^N(U) is usually rewritten using N - 2 clean ancilla qubits as a
    Toffoli compute ladder, one two-controlled U gate, and the reverse Toffoli
    ladder. If U is a tensor Pauli with weight at least 3, the pass instead
    computes the full conjunction of all N controls into N - 1 clean ancillas,
    applies one single-control Pauli gate per non-identity target, then
    uncomputes the conjunction. Open controls are handled by temporary X
    conjugations, matching Qiskit's ctrl_state convention where the rightmost
    control-state bit corresponds to the first control qubit in qargs.

    The returned circuit reuses the original circuit registers and appends one
    ancilla register sized for the largest expanded gate. Gates with fewer than
    three controls are copied unchanged.
    """
    max_ancillas = 0
    for instruction in qc.data:
        op = instruction.operation
        if isinstance(op, ControlledGate) and op.num_ctrl_qubits >= 3:
            pauli_data = _controlled_gate_single_qubit_pauli_data(op)
            if pauli_data is not None and len(pauli_data["ops"]) >= 3:
                max_ancillas = max(max_ancillas, int(op.num_ctrl_qubits) - 1)
            else:
                max_ancillas = max(max_ancillas, int(op.num_ctrl_qubits) - 2)

    expanded = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    expanded.global_phase = qc.global_phase
    expanded.metadata = None if qc.metadata is None else dict(qc.metadata)

    ancillas = []
    if max_ancillas > 0:
        existing_names = {reg.name for reg in qc.qregs + qc.cregs}
        reg_name = ancilla_name
        suffix = 0
        while reg_name in existing_names:
            suffix += 1
            reg_name = f"{ancilla_name}_{suffix}"
        ancilla_reg = QuantumRegister(max_ancillas, reg_name)
        expanded.add_register(ancilla_reg)
        ancillas = list(ancilla_reg)

    for instruction in qc.data:
        op = instruction.operation
        qargs = list(instruction.qubits)
        clbits = list(instruction.clbits)

        if not isinstance(op, ControlledGate) or op.num_ctrl_qubits < 3:
            expanded.append(op, qargs, clbits)
            continue

        if clbits:
            raise ValueError("Cannot expand classically-bit-attached controlled gates.")

        ctrl_count = int(op.num_ctrl_qubits)
        controls = qargs[:ctrl_count]
        targets = qargs[ctrl_count:]
        pauli_data = _controlled_gate_single_qubit_pauli_data(op)
        pauli_ops = None if pauli_data is None else pauli_data["ops"]
        use_full_control_pauli_expansion = pauli_ops is not None and len(pauli_ops) >= 3
        work_size = ctrl_count - 1 if use_full_control_pauli_expansion else ctrl_count - 2
        work = ancillas[:work_size]
        ctrl_bits = _ctrl_state_bits(op)

        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)

        if use_full_control_pauli_expansion:
            _append_full_control_pauli_expansion(
                expanded,
                controls,
                targets,
                work,
                pauli_ops,
                pauli_data["phase"],
            )
        else:
            expanded.ccx(controls[0], controls[1], work[0])
            for ctrl_index in range(2, ctrl_count - 1):
                expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])

            cc_u = op.base_gate.control(num_ctrl_qubits=2, ctrl_state="11")
            expanded.append(cc_u, [work[-1], controls[-1]] + targets)

            for ctrl_index in range(ctrl_count - 2, 1, -1):
                expanded.ccx(work[ctrl_index - 2], controls[ctrl_index], work[ctrl_index - 1])
            expanded.ccx(controls[0], controls[1], work[0])

        for ctrl, bit in zip(controls, ctrl_bits[::-1]):
            if bit == "0":
                expanded.x(ctrl)
    if if_optimize:
        expanded = optimize_expanded_circuit(expanded, ancillas)
        expand_large_controlled_gates.last_boolean_expressions = getattr(
            optimize_expanded_circuit,
            "last_boolean_expressions",
            [],
        )
    else:
        expand_large_controlled_gates.last_boolean_expressions = (
            encode_control_block_boolean_expressions(expanded, ancillas)
        )
    return expanded


expand_larger_controlled_gates = expand_large_controlled_gates


class BlockEncoding(_LegacyBlockEncoding):
    """Block-encoding with multi-candidate (k = w, w-1) basis selection."""

    def circuit(self, opt=None):
        if opt != 'Matrix-order':
            return super().circuit(opt=opt)

        self.mccount = 0
        self.cxcount = 0

        qc_u, tcount, mccount, cxcount, ctrl_size = self.mulplex_U_opt_order()
        qc_u = expand_larger_controlled_gates(qc_u)

        qc_select = self.mulplex_B_opt_order(ctrl_size)
        ctrl = QuantumRegister(ctrl_size, 'ctrl')
        sys = QuantumRegister(self.sys_size, 'sys')

        ancilla_size = qc_u.num_qubits - ctrl_size - self.sys_size
        if ancilla_size > 0:
            anc = QuantumRegister(ancilla_size, 'anc')
            qc = QuantumCircuit(ctrl, sys, anc, name='BlockEncoding')
            qc_u_qubits = list(ctrl) + list(sys) + list(anc)
        else:
            qc = QuantumCircuit(ctrl, sys, name='BlockEncoding')
            qc_u_qubits = list(ctrl) + list(sys)

        qc.compose(qc_select, qubits=ctrl, inplace=True)  # type: ignore
        qc.compose(qc_u, qubits=qc_u_qubits, inplace=True)
        qc.compose(qc_select.inverse(), qubits=ctrl, inplace=True)  # type: ignore

        self.tcount = tcount
        self.mccount += mccount
        self.cxcount = cxcount
        self.succ_prob = np.sum(self.coeff_list)
        self.circuit_width = qc.num_qubits
        return qc

    def find_optimal_order_matrices(self):
        pauli_list = []
        phase_list = []
        zero_phase = 0.0
        for i, ms in enumerate(self.mat_list):
            if isinstance(ms, tuple) or isinstance(ms, list):
                pauli_op, coeff = ms
                if pauli_op != 'I' * len(pauli_op):  # type: ignore
                    pauli_list.append(pauli_op)  # type: ignore
                    phase_list.append(coeff)
                else:
                    zero_phase = 2 * np.angle(coeff) / np.pi
        pauli_list = PauliList(pauli_list)
        x_part, z_part = pauli_list.x, pauli_list.z
        symplectic_matrix = np.hstack((x_part, z_part)).astype(int)
        phase_array = np.array([2 * np.angle(phase) / np.pi for phase in phase_list])

        sorted_matrix_z, phases_z = sort_symplectic_matrix(symplectic_matrix, phase_array, crit='z')
        sorted_matrix_x, phases_x = sort_symplectic_matrix(symplectic_matrix, phase_array, crit='x')

        M = sorted_matrix_z.shape[0]
        w = int(np.ceil(np.log2(len(self.coeff_list))))

        identity_label = None
        for ms in self.mat_list:
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            if pauli_op == 'I' * len(pauli_op):
                identity_label = pauli_op
                break

        def _pauli_weight(label):
            return sum(1 for c in label if c != 'I')

        def _build_candidate(k):
            ## Stage I: greedy basis on both x-prior and z-prior sortings.
            sel_z, cov_z, U_z = greedy_generator_selection(sorted_matrix_x, w, k)
            sel_x, cov_x, U_x = greedy_generator_selection(sorted_matrix_z, w, k)

            if cov_z > cov_x:
                cand_matrix = sorted_matrix_x
                cand_phases = phases_x
                cand_selected = sorted_matrix_x[sel_z]
                remained_indices = set(range(M)) - set(sel_z)
                cand_U = U_z
                cand_remaining = [sorted_matrix_x[i] for i in remained_indices
                                  if sorted_matrix_x[i].tobytes() not in cand_U]
            else:
                cand_matrix = sorted_matrix_z
                cand_phases = phases_z
                cand_selected = sorted_matrix_z[sel_x]
                remained_indices = set(range(M)) - set(sel_x)
                cand_U = U_x
                cand_remaining = [sorted_matrix_z[i] for i in remained_indices
                                  if sorted_matrix_z[i].tobytes() not in cand_U]

            ## Stages II/III: subspace + additional placement.
            subspace_modes = assign_subspace_modes(w, cand_selected, set(cand_U))
            cand_additional = assign_additional_modes(w, subspace_modes, cand_remaining)

            ## Identity handling: keep an explicit identity slot at ctrl = 0...0.
            if identity_label is not None:
                zero_ctrl = '0' * w
                used_ctrl_ids = {ctrl for _, ctrl in cand_additional}
                zero_idx = next((i for i, (_, c) in enumerate(cand_additional) if c == zero_ctrl), None)
                if zero_idx is None:
                    cand_additional.append((identity_label, zero_ctrl))
                else:
                    old_label, _ = cand_additional[zero_idx]
                    if old_label != identity_label:
                        displaced_mode = cand_additional[zero_idx]
                        new_ctrl = None
                        for ctrl_int in range(2 ** w):
                            candidate = bin(ctrl_int)[2:].zfill(w)
                            if candidate not in used_ctrl_ids:
                                new_ctrl = candidate
                                break
                        cand_additional[zero_idx] = (identity_label, zero_ctrl)
                        if new_ctrl is not None:
                            cand_additional.append((displaced_mode[0], new_ctrl))
            cand_additional = sorted(cand_additional, key=lambda x: int(x[1], 2))

            ## Möbius inversion gives g_l. Cost C = Σ w(ctrl) · w_s(g_l).
            cand_vec_phase_lookup = build_vec_phase_lookup(cand_matrix, cand_phases)
            cand_mobius = mobius_invert_modes_with_phases(
                w, cand_additional, cand_vec_phase_lookup, zero_phase
            )
            g_modes = cand_mobius['g_modes_with_phase']
            cost = sum(ctrl.count('1') * _pauli_weight(label) for label, ctrl, _ in g_modes)

            return {
                'k': k,
                'cost': cost,
                'additional_modes': cand_additional,
                'chosen_matrix': cand_matrix,
                'chosen_phases': cand_phases,
                'vec_phase_lookup': cand_vec_phase_lookup,
                'mobius_result': cand_mobius,
            }

        ## Two candidates: k = w and k = w - 1 (skip when w-1 < 1).
        candidate_ks = [w]
        if w - 1 >= 1:
            candidate_ks.append(w - 1)

        candidates = [_build_candidate(k) for k in candidate_ks]
        ## Tie-break: prefer larger k (closer to legacy behaviour) on equal cost.
        best = min(candidates, key=lambda c: (c['cost'], -c['k']))

        additional_modes = best['additional_modes']

        self.additional_modes = additional_modes
        self.candidate_costs = [(c['k'], c['cost']) for c in candidates]
        self.selected_k = best['k']

        ## Build coefficient list aligned with additional_modes order.
        coeff_pool_by_label = {}
        for idx, ms in enumerate(self.mat_list):
            if not isinstance(ms, tuple) and not isinstance(ms, list):
                continue
            pauli_op, _phase = ms
            coeff = self.coeff_list[idx]
            coeff_pool_by_label.setdefault(pauli_op, []).append(coeff)

        coeff_mode_dict = {}
        for pauli_label, ctrl_value in additional_modes:
            coeff_pool = coeff_pool_by_label.get(pauli_label, [])
            if len(coeff_pool) > 0:
                coeff_mode = coeff_pool.pop(0)
            else:
                coeff_mode = 0.0 + 0.0j
            coeff_mode_dict[ctrl_value] = coeff_mode

        self.coeff_mode_dict = coeff_mode_dict
        self.vec_phase_lookup = best['vec_phase_lookup']
        self.basis_modes_with_phase, self.nonbasis_modes_with_phase = extract_basis_modes_with_phases(
            additional_modes, self.vec_phase_lookup, zero_phase
        )
        self.mobius_phase_result = best['mobius_result']

        return coeff_mode_dict, w


def _test_complex_boolean_expression_encoding_example() -> None:
    from qiskit.circuit.library import XGate

    ctrl = QuantumRegister(5, 'ctrl')
    anc = QuantumRegister(3, 'anc')
    qc = QuantumCircuit(ctrl, anc, name='complex_boolean_encoding_demo')

    qc.ccx(ctrl[0], ctrl[2], anc[0])
    qc.append(XGate().control(2, ctrl_state='10'), [ctrl[0], ctrl[4], anc[1]])
    qc.ccx(ctrl[1], ctrl[3], anc[2])
    qc.ccx(ctrl[1], ctrl[2], anc[0])
    qc.ccx(ctrl[0], ctrl[4], anc[1])
    qc.ccx(ctrl[1], ctrl[3], anc[2])

    ancilla_set = set(anc)
    for instruction in qc.data:
        controls, targets = _instruction_controls_targets(instruction)
        if len(targets) == 1 and targets[0] in ancilla_set:
            assert len(controls) <= 2

    encoded_blocks = encode_control_block_boolean_expressions(qc, anc)

    anc0_label = _qubit_label(qc, anc[0])
    anc0_entry = next(entry for entry in encoded_blocks if entry['target'] == anc0_label)
    assert anc0_entry['permuted_gate_indices'] == [0, 3, 1, 4, 2, 5]
    assert anc0_entry['xor_expression'] == "(ctrl[0] & ctrl[2]) ^ (ctrl[1] & ctrl[2])"
    assert anc0_entry['simplified_expression'] == "(ctrl[0] ^ ctrl[1]) & ctrl[2]"
    assert anc0_entry['gate_sequence'] == [
        "CNOT(ctrl[0], ctrl[1])",
        "CCU(ctrl[1], ctrl[2], anc[0]; U=X)",
        "CNOT(ctrl[0], ctrl[1])",
    ]

    anc1_label = _qubit_label(qc, anc[1])
    anc1_entry = next(entry for entry in encoded_blocks if entry['target'] == anc1_label)
    assert anc1_entry['xor_expression'] == "(~ctrl[0] & ctrl[4]) ^ (ctrl[0] & ctrl[4])"
    assert anc1_entry['simplified_expression'] == "ctrl[4]"
    assert anc1_entry['gate_sequence'] == [
        "CNOT(ctrl[4], anc[1])",
    ]

    anc2_label = _qubit_label(qc, anc[2])
    anc2_entry = next(entry for entry in encoded_blocks if entry['target'] == anc2_label)
    assert anc2_entry['xor_expression'] == "(ctrl[1] & ctrl[3]) ^ (ctrl[1] & ctrl[3])"
    assert anc2_entry['simplified_expression'] == "0"
    assert anc2_entry['gate_sequence'] == []

    optimized = optimize_expanded_circuit(qc, anc)
    optimized_ops = [
        (
            instruction.operation.name,
            [_qubit_label(optimized, qubit) for qubit in instruction.qubits],
        )
        for instruction in optimized.data
    ]
    assert optimized_ops == [
        ("cx", ["ctrl[0]", "ctrl[1]"]),
        ("ccx", ["ctrl[1]", "ctrl[2]", "anc[0]"]),
        ("cx", ["ctrl[0]", "ctrl[1]"]),
        ("cx", ["ctrl[4]", "anc[1]"]),
    ]


def _test_high_weight_controlled_pauli_expansion_example() -> None:
    from qiskit.quantum_info import Pauli

    ctrl = QuantumRegister(4, 'ctrl')
    sys = QuantumRegister(3, 'sys')
    qc = QuantumCircuit(ctrl, sys, name='controlled_pauli_expansion_demo')
    qc_pauli = QuantumCircuit(3)
    qc_pauli.append(Pauli('XZZ'), range(3))
    qc_pauli = qc_pauli.decompose()
    ctrl_pauli = qc_pauli.to_gate().control(4, ctrl_state='1111')
    pauli_data = _controlled_gate_single_qubit_pauli_data(ctrl_pauli)
    assert pauli_data is not None
    assert pauli_data["ops"] == [("Z", 0), ("Z", 1), ("X", 2)]
    qc.append(ctrl_pauli, list(ctrl) + list(sys))

    expanded = expand_large_controlled_gates(qc)
    expanded_ops = [
        (
            instruction.operation.name,
            [_qubit_label(expanded, qubit) for qubit in instruction.qubits],
        )
        for instruction in expanded.data
    ]
    assert expanded_ops == [
        ("ccx", ["ctrl[0]", "ctrl[1]", "mc_anc[0]"]),
        ("ccx", ["mc_anc[0]", "ctrl[2]", "mc_anc[1]"]),
        ("ccx", ["mc_anc[1]", "ctrl[3]", "mc_anc[2]"]),
        ("cz", ["mc_anc[2]", "sys[0]"]),
        ("cz", ["mc_anc[2]", "sys[1]"]),
        ("cx", ["mc_anc[2]", "sys[2]"]),
        ("ccx", ["mc_anc[1]", "ctrl[3]", "mc_anc[2]"]),
        ("ccx", ["mc_anc[0]", "ctrl[2]", "mc_anc[1]"]),
        ("ccx", ["ctrl[0]", "ctrl[1]", "mc_anc[0]"]),
    ]
    assert expanded.num_qubits == 10


if __name__ == "__main__":
    _test_complex_boolean_expression_encoding_example()
    _test_high_weight_controlled_pauli_expansion_example()
