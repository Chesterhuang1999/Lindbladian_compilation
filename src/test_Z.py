import numpy as np
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.library import save_density_matrix, save_statevector

from series_expansion import block_encoding_matrixsum, lcu_prepare_tree
from channel_IR import * 
from scipy.linalg import expm


if __name__ == "__main__":
    H = [('ZZI', -1), ('IZZ', -1), ('ZIZ', -1),('XII', -1), ('IXI', -1), ('IIX', -1)]
    gamma = np.sqrt(0.1)/2 
    L_list = [[('XII', gamma), ('YII', -1j * gamma)], [('IXI', gamma), ('IYI', -1j * gamma)], [('IIX', gamma), ('IIY', -1j * gamma)]]
    delta_t = 0.1
    TFIM_lind = Lindbladian(H, L_list)
    
    ini_state = Statevector.from_label('+00')
    H_ms, L_ms_list = TFIM_lind.H, TFIM_lind.L_list

    H_circ = block_encoding_matrixsum(H_ms)
    L_circ_list = [block_encoding_matrixsum(L_m) for L_m in L_ms_list]
    ## Prepare baseline
    A0 = Operator(H_ms.eff_op())
    Al_list = [A0 @ Operator(L_m.eff_op()) for L_m in L_ms_list]
    sum_op = matsum_mul(H_ms.adj(), H_ms)
    ## Prepare kraus coeffs 
    h_coeff = H_ms.pauli_norm()
    l_coeffs = [h_coeff * L_m.pauli_norm() for L_m in L_ms_list]

    for L_m in L_ms_list:
        pass

