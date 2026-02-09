from channel_IR import * 
from subroutine import * 
from block_encoding import BlockEncoding
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix, partial_trace
from qiskit.circuit.library import StatePreparation




class channels:
    """
    A intermediate representation for the (probabilistic ensemble) of quantum channels.
    The channels are of the form: [(s_j, E_j)], where s_j is the probability weight and 
    E_j = sum_l A_jk\rho A_jk^dagger is the j-th quantum channel in the ensemble. 
    """

    def __init__(self, Lind: Lindbladian, choices : str, delta_t: float):
        self.Lind = Lind
        self.Hamiltonian = self.Lind.H
        self.L_list = self.Lind.L_list
        self.t = delta_t
        self.choices = choices
        self.sys_size = self.Hamiltonian.size
        self.channels = self.to_channels()

    def to_channels(self):
        if self.choices == 'basic':
            self.to_channel_basic_LCU()
        elif self.choices == 'series':
            K = 2
            K1 = 2
            q = 4
            self.to_channel_series(K, q,  K1)
        elif self.choices == 'trajectory':
            self.to_channel_trajectory()
        

    def to_channel_basic_LCU(self):
        sqdelta = np.sqrt(self.t)
        channel = []
        iden = PauliAtom('I' * self.Hamiltonian.size, phase = 1.0)
        kraus_0_basis = Matrixsum([(iden, 1.0)])

        H_copy = deepcopy(self.Hamiltonian)
        H_copy.mul_coeffs(-1j* self.t)
        kraus_0_basis = kraus_0_basis.add(H_copy)

        Ls_copy = []
        for Ls in self.L_list:
            Ls_copy.append(deepcopy(Ls))
        for i in range(len(Ls_copy)):
            Ls_product = matsum_mul(Ls_copy[i].adj(), Ls_copy[i])
            Ls_product.mul_coeffs(-0.5 * self.t)
            kraus_0_basis = kraus_0_basis.add(Ls_product)
        
        channel.append(kraus_0_basis)
        for Ls in self.L_list:
            Ls_copy_2 = deepcopy(Ls)
            Ls_copy_2.mul_coeffs(sqdelta)
            channel.append(Ls_copy_2)

        ### Calculate the success prob here
        coeff_sq = 0.0
        coeff_sq = np.sum([ms.pauli_norm()**2 for ms in channel])
        success_prob_th = 1 / coeff_sq
        self.succ_prob = success_prob_th

        self.select_size = len(channel)
        self.ctrl_size = max([len(ms.instances) for ms in channel])
        # c = channel_ensemble([channels])
        
        self.channels = []
        self.channels.append([channel])
    def create_single_channel(self, k: int, index: list, q :int, K1:int,
                             eff_H : Matrixsum):
        kraus_ms = []
        l_list = index[1: k + 1]
        j_list = index[k + 1: 2 * k + 1]
        coeff_sum_index = 1.0 ## Set coeff sum value

        ## Generate quadrature points and weights according to a biased Legendre function
        points, weights = quadrature_points_N_weights(k, q, self.t, j_list)
        points.append(0) ## Add minimum point 0
        weights_prod = np.prod(weights)

        ## Mul coefficient 'sqrt(w_(jk) * ... * w_(jk,...j1))'
        coeff_sum_index *= np.sqrt(weights_prod)
        # ctrl_size = []

        for j in range(k + 1):
            if j == k: 
                delta_t = self.t - points[0]
            else:
                delta_t = points[k - j - 1] - points[k - j]
            H_eff_series, coeff_sum_coh = exp_term_expansion(eff_H, K1, delta_t)
            coeff_sum_index *= coeff_sum_coh
            kraus_ms.append(H_eff_series)
            # ctrl_size.append(H_eff_series.ctrl_size)
            if j < k:
                L_j = self.L_list[l_list[k - j - 1]]
                kraus_ms.append(L_j)
                # ctrl_size.append(L_j.ctrl_size)
                coeff_sum_index *= L_j.pauli_norm()
        return kraus_ms, coeff_sum_index
    def to_channel_series(self, K: int, q: int, K1: int):
        H_eff = self.Lind.effective_H()
        m = len(self.Lind.L_list)
        coeff_sums = []
        H_eff_series0, coeff_sum_series0  = exp_term_expansion(H_eff, K1, self.t)
        coeff_sums.append(coeff_sum_series0)
        # ctrl_sizes.append([H_eff_series0.ctrl_size])
        # ctrl_sizes_max = H_eff_series0.ctrl_size
        ## In higher order series expansion method, channels is list of lists
        # (each op in the first level is a list of matrixsums)
        self.channels = []
        self.channels.append([H_eff_series0])
        for k in range(1, K + 1):
            index_set_k = create_index_set(k, q, m)
            sum_of_kraus += len(index_set_k)
            for index in index_set_k:
                channel_ops, coeff_sum_index = self.create_single_channel(k, index, q, K1, H_eff)
                # ctrl_sizes.append([ctrl_size])
                # ctrl_sizes_max = max(max(ctrl_size), ctrl_sizes_max)
                self.channels.append(channel_ops)
                coeff_sums.append(coeff_sum_index)
        # ctrl_size_max = max(ctrl_sizes)
        # self.ctrl_size = ctrl_sizes
        self.coeff_sums = coeff_sums
        
        
    def to_channel_trajectory(self):
        self.channels = []
    def channels_to_circuit(self):
        if self.choices == 'basic':
            qc, qubit_regs = self.to_circuit_basic_LCU()
        elif self.choices == 'series':
            qc, qubit_regs = self.to_circuit_series()

        self.circuit = qc
        self.qubit_regs = qubit_regs
        return qc, qubit_regs
    def to_circuit_basic_LCU(self):
        channel = self.channels
        coeff_sums = [ms.pauli_norm() for ms in channel]
        ctrl = QuantumRegister(self.ctrl_size, 'c')
        select = QuantumRegister(self.select_size, 's')
        sys = QuantumRegister(self.sys_size, 'sys')
        qc = QuantumCircuit(select, ctrl, sys)
        ## Prepare the superposition state 
        LCU_ini = prep_sup_state(coeff_sums)
        qc.compose(LCU_ini, qubits=list(range(self.select_size)), inplace=True) 

        ## First construct elementary block encodings then select them using select register. 
        for i, ms in enumerate(channel):
            ctrl_value = bin(i)[2:].zfill(self.select_size)
            qc_be_ms = BlockEncoding(ms).circuit()
            ctrl_size_ms = qc_be_ms.num_qubits - self.sys_size
            U_be_ms = qc_be_ms.to_gate().control(num_ctrl_qubits=self.select_size, ctrl_state=ctrl_value)
            qc.append(U_be_ms, qargs = list(select) + list(ctrl[:ctrl_size_ms]) + list(sys))
        
        qc.save_statevector('final_state') #type: ignore

        qubit_regs = [ctrl, select, sys]
        return [qc, qubit_regs]
    def to_circuit_series(self):
        assert isinstance(self.ctrl_size, list)
        assert self.choices == 'series' 

        channel = self.channels
        ctrl_size_max = 0
        ctrl_sizes_channels = []
        for j, ops_list in enumerate(channel):
            
            ctrl_sizes = [ms.ctrl_size for ms in ops_list]
            ctrl_size_max = max(np.sum(ctrl_sizes), ctrl_size_max)
            ctrl_sizes_channels.append(ctrl_sizes)
        sel_size = self.select_size
        sel = QuantumRegister(sel_size, 'sel')
        LCU_ini = prep_sup_state(self.coeff_sums)
        ctrl = QuantumRegister(ctrl_size_max, 'ctrl')
        sys = QuantumRegister(self.sys_size, 'sys')
        qc_main = QuantumCircuit(sel, ctrl, sys)
        
        qc_main.compose(LCU_ini, qubits = sel, inplace = True)

        for j, ms_list in enumerate(channel):
            ctrl_sizes = ctrl_sizes_channels[j]
            ctrl_size_j = np.sum(ctrl_sizes)
            qubit_count_j = ctrl_size_j + self.sys_size
            qc_j = QuantumCircuit(qubit_count_j)
            ctrl_offset = 0
            for k, ms in ops_list:
                qc = BlockEncoding(ms).circuit()
                qc_j.compose(qc, qubits = list(range(ctrl_offset, ctrl_offset + ctrl_sizes[k])) + list(range(ctrl_size_j, qubit_count_j)))
                ctrl_offset += ctrl_sizes[k]
            sel_value = bin(j)[2:].zfill(sel_size)
            qc_j.to_gate().control(num_ctrl_qubits = sel_size, ctrl_state= sel_value)
            qc_main.append(qc_j, qargs = qc_main.qubits[:sel_size] + qc_main.qubits[sel_size:sel_size + ctrl_size_j] 
                    + qc_main.qubits[sel_size + ctrl_size_max:])
        
        qubit_regs = [sel, ctrl, sys]
       
        return qc_main, qubit_regs
    def get_gate_counts(self):
        pass
    def get_block_counts(self): 
        self.to_channels()
        
        channel = self.channels 
        if self.channels is None or len(self.channels) == 0:
            return 0
        block_count = 0
        for channel in self.channels:
            for ms in channel: 
                block_count += ms.length
        return block_count
    def get_mcc_counts(self):
        pass
    def get_qubit_width(self):

        self.channels_to_circuit()

        system_size = self.qubit_regs[-1]

        return np.sum(self.qubit_regs) - system_size

