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
        
        t_count_per_ctrl = 4 
        cx_count_per_ctrl = 4
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
        ### For test: genearte the order for matrices
        ctrlv_dict = {"ZII": "0100", "IZI": "0010", "IIZ": "0001", "ZZI": "0110", "ZIZ": "0101", "IZZ": "0011", "XII": "1100", "IXI": "1010", "IIX": "1001"}
        for i, ms in enumerate(mat_list):

            if isinstance(ms, SparsePauliOp):
                pauli_op, phase = ms.paulis[0], ms.coeffs[0]
                label = pauli_op.to_label()
                qc_pauli = QuantumCircuit(pauli_op.num_qubits)
                qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
                qc_pauli.global_phase = np.angle(phase)
                qc_pauli = qc_pauli.decompose()
                U_elem = qc_pauli.to_gate()
                
            else:
                if ms.shape[0] < 2**sys_size:
                    pad_size = 2**sys_size // ms.shape[0]
                    ms = np.kron(ms, np.eye(pad_size))
            
            # control_values =  bin(i)[2:].zfill(ctrl_size) 
            control_values = ctrlv_dict.get(label)
            if len(qc_pauli.data) > 0 : ### Identity is ignored
                ctrl_U_elem = U_elem.control(num_ctrl_qubits = ctrl_size, ctrl_state = control_values)
                qc.append(ctrl_U_elem, range(ctrl_size + sys_size))
                mccount += (ctrl_size - 1) * len(qc_pauli.data)

        tcount = mccount * t_count_per_ctrl
        cxcount = mccount * cx_count_per_ctrl
        return qc, tcount, mccount, cxcount
    def mulplex_U_opt(self, mat_list, ctrl_size, sys_size):
        from qiskit.circuit.library import XGate
        mccount = 0
        tcount = 0
        cxcount = 0
        t_counts_per_ccx = 4
        cx_counts_per_ccx = 4  
        opt_circuit = QuantumCircuit(1 + 2 * ctrl_size + sys_size)
        opt_circuit.x(0)
        sel_regs = [2 * j + 1 for j in range(ctrl_size)]
        anc_regs = [2 * j + 2 for j in range(ctrl_size)]
        def apply_left_enc(j):
            opt_circuit.reset(anc_regs[j])
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            top = 0 if j == 0 else anc_regs[j - 1]
            opt_circuit.append(ccxgate_c, [top, sel_regs[j], anc_regs[j]])
        def apply_right_enc(j):
            ctrl_bval = control_values[j]
            ccxgate_c = XGate().control(num_ctrl_qubits = 2, ctrl_state = ctrl_bval + '1')
            top = 0 if j == 0 else anc_regs[j - 1]
            opt_circuit.append(ccxgate_c, [top, sel_regs[j], anc_regs[j]])
            opt_circuit.reset(anc_regs[j])
        maxctrl_value = bin(len(mat_list) - 1)[2:].zfill(ctrl_size)
        def find_bit_to_remove(max_val, cur_val):
            candidate_bits = []
            for j in range(ctrl_size):
                if max_val[j] != cur_val[j]:
                    return candidate_bits
                elif max_val[j] == '0' and cur_val[j] == '0':
                    candidate_bits.append(j)
                else:
                    continue
            return candidate_bits
        for i, mat in enumerate(mat_list):
            
            pauli_op, phase = mat.paulis[0], mat.coeffs[0]
            qc_pauli = QuantumCircuit(pauli_op.num_qubits)
            qc_pauli.append(pauli_op, range(pauli_op.num_qubits)) #type: ignore
            qc_pauli.global_phase = np.angle(phase)
            qc_pauli = qc_pauli.decompose()
            numq = qc_pauli.num_qubits
            control_values = bin(i)[2:].zfill(ctrl_size)
            index_remove = find_bit_to_remove(maxctrl_value, control_values)

            if i == 0:
                cval_next = bin(1)[2:].zfill(ctrl_size)   
                for j in range(ctrl_size):
                    apply_left_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size,2 * ctrl_size + 1 +  numq)))
                opt_circuit.cx(anc_regs[ctrl_size - 2], anc_regs[ctrl_size - 1])
                cxcount += 1
            elif i == len(mat_list) - 1:
                cval_prev = bin(i - 1)[2:].zfill(ctrl_size)
                diff_prev = next(j for j in range(ctrl_size) if cval_prev[j] != control_values[j])
                if diff_prev != ctrl_size - 1:
                    for j in range(diff_prev + 1, ctrl_size):
                        apply_left_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size,2 * ctrl_size + 1 + numq)))
                for j in range(ctrl_size - 1, -1, -1):
                    apply_right_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
            else:
                cval_prev, cval_next = bin(i - 1)[2:].zfill(ctrl_size), bin(i + 1)[2:].zfill(ctrl_size)
                ## Find the first bit that differs
                diff_prev = next(j for j in range(ctrl_size) if cval_prev[j] != control_values[j])
                ## Apply left encodings from diff_prev to the end
                if diff_prev != ctrl_size - 1:
                    for j in range(diff_prev + 1, ctrl_size):
                        apply_left_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                ## Apply the controlled circuit
                opt_circuit.append(qc_pauli.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * ctrl_size,2 * ctrl_size + 1 + numq)))
                diff_next = next(j for j in range(ctrl_size) if cval_next[j] != control_values[j])
                ## Apply right encodings from diff_next to the end

                if diff_next != ctrl_size - 1:
                    for j in range(ctrl_size - 1, diff_next, -1):
                        apply_right_enc(j)
                        mccount += 1
                        tcount += t_counts_per_ccx
                        cxcount += cx_counts_per_ccx
                ## Apply a CX to flip the differed bit
                if diff_next != 0:
                    opt_circuit.cx(anc_regs[diff_next - 1], anc_regs[diff_next])
                    cxcount += 1
                else:
                    opt_circuit.cx(0, anc_regs[0])
        return opt_circuit, tcount, mccount, cxcount
    @staticmethod
    def _count_ctrl_qubits(qc: QuantumCircuit):
        cxcount = 0
        mccount = 0
        for inst, _, _  in qc.data:
            if isinstance(inst, ControlledGate):
                if inst.num_ctrl_qubits == 1:
                    cxcount += 1
                else:
                    mccount += inst.num_ctrl_qubits - 1
            else:
                cxcount += getattr(inst, 'num_ctrl_qubits', 0)
        return cxcount, mccount
    def mulplex_B(self, coeff_list, ctrl_size):
        cx_per_toffoli = 4
        sum_coeff = sum([abs(c) for c in coeff_list])
        norm_coeffs = [abs(c)/sum_coeff for c in coeff_list]
        probs = np.zeros(2**ctrl_size, dtype = float)
        amps = np.zeros(2**ctrl_size, dtype = float)
        for i, nc in enumerate(norm_coeffs):
            probs[i] = nc
            amps[i] = np.sqrt(nc)
        
        qc = lcu_prepare_tree(probs) 
        # cxcount, mccount = self._count_ctrl_qubits(qc)
        # if self.mccount == 0:
        #     self.mccount = mccount
        # else:
        #     self.mccount += mccount
        # self.cxcount = cxcount if self.cxcount == 0 else self.cxcount + cxcount
        # self.cxcount += self.mccount * cx_per_toffoli
        return qc #type: ignore
    def circuit(self, opt = False):
        """
        Returns the block-encoding QuantumCircuit for the operator J.
        """
        self.mccount = 0
        self.cxcount = 0
        if self.ctrl_size == 0:
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(sys)
            if opt == False:
                qc_u, tcount, mccount, cxcount = self.mulplex_U(self.mat_list, 0, self.sys_size)
            else:
                qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, 0, self.sys_size)
            qc.compose(qc_u, qubits=sys[:], inplace=True)
            return qc
        
        if opt == False:
            qc_u, tcount, mccount, cxcount = self.mulplex_U(self.mat_list, self.ctrl_size, self.sys_size)
            
            ctrl = QuantumRegister(self.ctrl_size, 'ctrl')
            sys = QuantumRegister(self.sys_size, 'sys')
            qc = QuantumCircuit(ctrl, sys)
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            qc.compose(qc_select, qubits=ctrl, inplace=True) #type: ignore
            qc.compose(qc_u, qubits=qc.qubits, inplace=True)
            qc.compose(qc_select.inverse(), qubits=ctrl, inplace=True) #type: ignore
        else:
            qc_u, tcount, mccount, cxcount = self.mulplex_U_opt(self.mat_list, self.ctrl_size, self.sys_size)
            qc_select = self.mulplex_B(self.coeff_list, self.ctrl_size)
            ctrl_index = [2 * j + 1 for j in range(self.ctrl_size)]
            print(qc_u.draw())
            qc = QuantumCircuit(qc_u.num_qubits, name = "BlockEncoding")
            qc.compose(qc_select, qubits = ctrl_index, inplace = True) #type: ignore 
            qc.compose(qc_u, qubits = qc.qubits, inplace = True)
            qc.compose(qc_select.inverse(), qubits = ctrl_index, inplace = True) #type: ignore

        self.tcount = tcount
        self.mccount += mccount
        self.cxcount = cxcount
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

if __name__ == "__main__":
    from channel_LCU import Lindblad_to_channel 
    from qiskit.circuit.library import YGate
    N = 3
    H = []
    L_list = []
    gamma = np.sqrt(0.1)/2 
    for i in range(N):
        iden_str = 'I' * N
        Z_ind = [i, (i + 1) % N]
        Z_str = ''.join(['Z' if j in Z_ind else 'I' for j in range(N)])
        H.append((Z_str, -1))
        X_str = ''.join([('X' if j == i else 'I') for j in range(N)]) 
        H.append((X_str, -1))
        Y_str = ''.join([('Y' if j == i else 'I') for j in range(N)])
        L_list.append([(X_str, gamma), (Y_str, -1j * gamma)])
    # H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    
    # L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    delta_t = 0.1
    TFIM_lind = Lindbladian(H, L_list)

    channel_Lind, success_prob_th, coeff_sum = Lindblad_to_channel(TFIM_lind, delta_t)

    channel_Lind = channel_Lind.channels[0][1]
    ms = channel_Lind[0]
    ## Unoptimized version
    print(ms)
    ms_be = BlockEncoding(ms)

    print("Unoptimized version:")
    qc_be = ms_be.circuit(opt = False)   
    tcount, mccount, cxcount = ms_be.tcount, ms_be.mccount, ms_be.cxcount
    print(f"T-count: {tcount}, Multi-controlled gate count: {mccount}, CX count: {cxcount}")

    ## Optimized version I: unary iteration
    ms_be = BlockEncoding(ms)

    qc_be_opt1 = ms_be.circuit(opt = True)
    tcount_opt1, mccount_opt1, cxcount_opt1 = ms_be.tcount, ms_be.mccount, ms_be.cxcount
    print("Optimized version I (unary iteration):")
    print(f"T-count: {tcount_opt1}, Multi-controlled gate count: {mccount_opt1}, CX count: {cxcount_opt1}")

    ## Optimized version II: optimization over gate structures

    qc = QuantumCircuit(7)

    mccount, tcount, cxcount = 0, 0, 0
    L = ms.length
    w = int(np.ceil(np.log2(L)))
    for i in range(N):
        ## Implement -Z on first three control lines
        qc.p(np.pi, i)
        qc.cz(i, i + w)
        cxcount += 1
        ## Implement -Y on (0, 3), (1, 3), (2, 3)
    for i in range(N):
        qc.cp(np.pi, i, w - 1)
        ccy = YGate().control(num_ctrl_qubits = 2, ctrl_state = '11')
        qc.append(ccy, [i, w - 1, i + w])
        mccount += 1
        tcount += 4
        cxcount += 4
    print("Optimized version II: exploring Pauli structures in LCU")
    print(f"T-count: {tcount}, Multi-controlled gate count: {mccount}, CX count: {cxcount}")
        