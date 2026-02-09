from channel_IR import *
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, partial_trace
from qiskit.circuit.library import StatePreparation
from subroutine import lcu_prepare_tree, count_multiq_gates
import numpy as np
from qiskit_aer import AerSimulator
from qiskit.circuit.controlledgate import ControlledGate
class BlockEncoding:
    """
    Block Encoding Class for a given Matrixsum Operator J.
    Constructs the block-encoding circuit for the operator and provides
    resource estimation such as ancilla qubit usage and multi-qubit gate counts.
    """
    def __init__(self, J: Matrixsum):
        self.J = J
        self.coeff_list = [coeff for _, coeff in J.instances]
        self.mat_list = []
        for matrix, _ in J.instances:
            if isinstance(matrix, PauliAtom):
                self.mat_list.append(SparsePauliOp([matrix.expr], np.array([matrix.phase])))
            else:
                self.mat_list.append(matrix.to_operator().data)
        self.ctrl_size = int(np.ceil(np.log2(len(self.coeff_list))))
        self.sys_size = J.size
    def mulplex_U(self, mat_list, ctrl_size, sys_size):
        t_count_per_toffoli = 7
        mccount = 0
        if ctrl_size == 0:
            qc = QuantumCircuit(sys_size)
            assert len(mat_list) == 1
            pauli_op, phase = mat_list[0].paulis[0], mat_list[0].coeffs[0]
            qc_pauli = QuantumCircuit(pauli_op.num_qubits)
            qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
            qc_pauli.global_phase = np.angle(phase)
            qc_pauli = qc_pauli.decompose()
            return qc_pauli
    
        qc = QuantumCircuit(ctrl_size + sys_size)
        
        for i, ms in enumerate(mat_list):
            if isinstance(ms, SparsePauliOp):
                pauli_op, phase = ms.paulis[0], ms.coeffs[0]
                qc_pauli = QuantumCircuit(pauli_op.num_qubits)
                qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
                qc_pauli.global_phase = np.angle(phase)
                qc_pauli = qc_pauli.decompose()
                U_elem = qc_pauli.to_gate()
            else:
                if ms.shape[0] < 2**sys_size:
                    pad_size = 2**sys_size // ms.shape[0]
                    ms = np.kron(ms, np.eye(pad_size))
            control_values =  bin(i)[2:].zfill(ctrl_size) 

            if len(qc_pauli.data) > 0 : ### Identity is ignored
                ctrl_U_elem = U_elem.control(num_ctrl_qubits = ctrl_size, ctrl_state = control_values)
                qc.append(ctrl_U_elem, range(ctrl_size + sys_size))
                mccount += 2 * (ctrl_size - 2 ) + 1

        tcount = mccount * t_count_per_toffoli
        return qc, tcount, mccount
    @staticmethod
    def _count_ctrl_qubits(qc: QuantumCircuit):
        total = 0
        for inst, _, _  in qc.data:
            if isinstance(inst, ControlledGate):
                total += inst.num_ctrl_qubits
            else:
                total += getattr(inst, 'num_ctrl_qubits', 0)
        return total
    def mulplex_B(self, coeff_list, ctrl_size):
        sum_coeff = sum([abs(c) for c in coeff_list])
        norm_coeffs = [abs(c)/sum_coeff for c in coeff_list]
        probs = np.zeros(2**ctrl_size, dtype = float)
        amps = np.zeros(2**ctrl_size, dtype = float)
        for i, nc in enumerate(norm_coeffs):
            probs[i] = nc
            amps[i] = np.sqrt(nc)
        
        qc = lcu_prepare_tree(probs) 
        if self.mccount == 0:
            self.mccount = self._count_ctrl_qubits(qc)
        else:
            self.mccount += self._count_ctrl_qubits(qc)  
        return qc #type: ignore
    def circuit(self):
        """
        Returns the block-encoding QuantumCircuit for the operator J.
        """
        if self.ctrl_size == 0:
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(sys)
            qc_u, tcount, mccount = self.mulplex_U(self.mat_list, 0, self.sys_size)
            qc.compose(qc, qubits=sys[:], inplace=True)
            return qc
        
        ctrl = QuantumRegister(self.ctrl_size, 'ctrl')
        sys = QuantumRegister(self.sys_size, 'sys')
        qc = QuantumCircuit(ctrl, sys)
        self.mccount = 0
        qc_u, tcount, mccount = self.mulplex_U(self.mat_list, self.ctrl_size, self.sys_size)
        qc.compose(self.mulplex_B(self.coeff_list, self.ctrl_size), qubits=ctrl, inplace=True) #type: ignore
        qc.compose(qc_u, qubits=qc.qubits, inplace=True)
        qc.compose(self.mulplex_B(self.coeff_list, self.ctrl_size).inverse(), qubits=ctrl, inplace=True) #type: ignore
        self.tcount = tcount
        self.mccount += mccount
        self.succ_prob = np.sum(self.coeff_list)
        return qc

    def pauli_norm(self):
        """
        Returns the sum of coefficients (success probability).
        """
        return self.J.pauli_norm()

    def resource_counts(self):
        """
        Returns a dictionary with resource estimates: ancilla qubits and multi-qubit gate counts.
        """
        qc = self.circuit()
        multiq, tcount = count_multiq_gates(qc)
        return {
            "ancilla_qubits": self.ctrl_size,
            "multi_controlled_gates": self.mccount,
            "t_gates": self.tcount,
        }
    

class AlgCircuitSimulator:
    """
    Base class: 
    Simulator for the algorithmic circuits 
    designed by block-encoding and LCU methods. 
    This class can be used to simulate the final statevector or density matrix,
    and find the final density matrix on system qubits according to the register sizes. 
    """
    def __init__(self, circuit: QuantumCircuit, reg_sizes: list[int]):
        self.circuit = circuit
        self.reg_sizes = reg_sizes
        self.transpiled_circuit = None

    def simulate(self, *args, **kwargs):
        raise NotImplementedError("simulate must be implemented in subclasses.")
    def transpile_circuit(self, *args, **kwargs):
        raise NotImplementedError("transpile_circuit must be implemented in subclasses.")


class AlgCircuitSVSimulator(AlgCircuitSimulator):
    def transpile_circuit(self, gate_class: list[str] | None = None, optimization_level: int = 1):
        backend = AerSimulator(method="statevector")
        self.transpiled_circuit = transpile(
            self.circuit,
            backend=backend,
            basis_gates=gate_class,
            optimization_level=optimization_level,
        )
        multiq, tcount = count_multiq_gates(self.transpiled_circuit)
        return self.transpiled_circuit, {"multi_qubit_gates": multiq, "t_gates": tcount}


    def simulate(self, initial_state: Statevector | None = None):
        qc = self.transpiled_circuit or self.circuit
        if initial_state is None:
            return Statevector.from_instruction(qc)
        qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
        qc_sim.initialize(initial_state, range(len(initial_state)))  # type: ignore
        qc_sim.compose(qc, qc.qubits, qc.clbits, inplace=True)
        simulator = AerSimulator(method="statevector")

        self.result_sv = simulator.run(qc_sim, shots = 1).result().data['final_state']


    def purification_sys(self):
        sv = self.result_sv
        total_dens = DensityMatrix(sv)
        sel_size, ctrl_size, sys_size = self.reg_sizes
        proj_0 = Operator.from_label('0' * ctrl_size) ## ctrl_register must be 0
        idenf = lambda x: Operator.from_label('I' * x)
        proj_full = idenf(sys_size).tensor(proj_0).tensor(idenf(sel_size))
        projected_dens = DensityMatrix(np.array(proj_full @ total_dens @ proj_full))
        system_dens = partial_trace(projected_dens, list(range(sel_size + ctrl_size)))
        self.dens_sys = system_dens 

        return system_dens


        
class AlgCircuitTNSimulator(AlgCircuitSimulator):
    def transpile_circuit(self, gate_class: list[str] | None = None, optimization_level: int = 1):
        backend = AerSimulator(method="matrix_product_state")
        self.transpiled_circuit = transpile(
            self.circuit,
            backend=backend,
            basis_gates=gate_class,
            optimization_level=optimization_level,
        )
        multiq, tcount = count_multiq_gates(self.transpiled_circuit)
        return self.transpiled_circuit, {"multi_qubit_gates": multiq, "t_gates": tcount}

    def simulate(self, initial_state: Statevector | None = None, bond_dim: int = 64):
        qc = self.transpiled_circuit or self.circuit
        simulator = AerSimulator(method="matrix_product_state")
        simulator.set_options(matrix_product_state_max_bond_dimension=bond_dim)
        qc_sim = QuantumCircuit(qc.qubits, qc.clbits)
        if initial_state is not None:
            qc_sim.initialize(initial_state)
        qc_sim.compose(qc, qc.qubits, qc.clbits, inplace=True)
        qc_sim.save_density_matrix(label="final_dm")  # type: ignore
        qc_sim = transpile(qc_sim, simulator, optimization_level=1)
        result = simulator.run(qc_sim, shots=1).result()
        return DensityMatrix(result.data()["final_dm"])
    


class Channels: 
    pass