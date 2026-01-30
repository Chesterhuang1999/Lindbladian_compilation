import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli

class basic_unitary():
    """
    A basic unitary class representing a_j U_j. 
    Params: 
    expr: a Pauli string or a unitary matrix
    coeff: coefficient before expr
    """
    def __init__(self, expr, coeff: complex, re = 0., im = 0.):
        abs_coeff = abs(coeff)
        if abs_coeff > 0: 
            phase = coeff / abs_coeff
        else: 
            phase = 1.0
        self.coeff = abs_coeff
        self.expr = None
        self.type = None
        if isinstance(expr, str) or isinstance(expr, Pauli) :
            self.expr = SparsePauliOp(expr, np.array([phase])) #type: ignore
            self.mat = self.expr.to_matrix()
            self.type = 'Pauli'   
        elif isinstance(expr, SparsePauliOp):
            self.expr = SparsePauliOp(expr.paulis, expr.coeffs * phase)
            self.mat = self.expr.to_matrix()
            self.type = 'Pauli'
       
        elif isinstance(expr, np.ndarray):
            self.mat = phase * expr
            self.type = 'Unitary'
       
        if re != 0.: 
            self.real = re
            self.coeff += re 
        else: 
            self.re = np.real(coeff)
        if im != 0.:
            self.im = im
            self.coeff += 1j * im 
        else:
            self.imag = np.imag(coeff)


class Matrixsum():
    """
    Linear combination of basic_unitary matrices. 
    """
    def __init__(self, mat_list, coeff_list):
        self.mat_list = []
        assert len(mat_list) == len(coeff_list), "Length of pauli list and coeff list must be the same."
        for i in range(len(mat_list)):
            mat = basic_unitary(mat_list[i], coeff = coeff_list[i])
            self.mat_list.append(mat)
    
class Lindbladian():
    """
    A Lindbladian operator class. 
    Parameters:
       H: Hamiltonian in either matrix form or Pauli sum form
       L_list: List of Lindblad operators in either matrix form or Pauli sum form
    """
    def __init__(self, H, L_list):
        ### Matrix input, lower to Paulisum form
        if isinstance(H, np.ndarray):
            H_pl = SparsePauliOp.from_operator(Operator(H))
            H_pl = H_pl.simplify(atol=1e-12)
            self.H = Matrixsum(H_pl.paulis, H_pl.coeffs)
        else:
            ## Expressed in Matrix sum form already
            self.H = Matrixsum(H[0], H[1])

        for L in L_list: 
            if isinstance(L, np.ndarray):
                L_pl = SparsePauliOp.from_operator(Operator(L))
                L_pl = L_pl.simplify(atol=1e-12)
                if not hasattr(self, 'L_list'):
                    self.L_list = []
            
                    
                self.L_list.append(Matrixsum(L_pl.paulis, L_pl.coeffs))
            else:
                ## Expressed in Matrix sum form already
                if not hasattr(self, 'L_list'):
                    self.L_list = []
                self.L_list.append(Matrixsum(L[0], L[1]))
        
        ## Input in the LCU form, store matrices and coeffs
        # elif form == 'lcu_mat':
        #     self.H = Matrixsum(H[0], H[1])
        #     for L in L_list:    
        #         if not hasattr(self, 'L_list'):
        #             self.L_list = []
        #         self.L_list.append(Matrixsum(L[0], L[1]))
        ## Directly input in the Paulisum form
        # elif form == 'pauli':
            
        #     self.H = Matrixsum(H[0], H[1])
        #     for L in L_list:
        #         if not hasattr(self, 'L_list'):
        #             self.L_list = []
        #         self.L_list.append(Matrixsum(L[0], L[1]))
        # else:
        #     raise ValueError("Form must be either 'mat' or 'pauli'.")

class channel_ensemble(): 
    """
    A intermediate representation for the (probabilistic ensemble) of quantum channels.
    The channels are of the form: [(s_j, E_j)], where s_j is the probability weight and 
    E_j = sum_l A_jk\rho A_jk^dagger is the j-th quantum channel in the ensemble. 
    """

    def __init__(self, channels, probs = None):
        """
        Initialize the channel_IR object.
        
        Args:
            channels (list of Matrixsum): A list of quantum channels, where each channel is represented as a list of Kraus operators.
            probs (list, optional): A list of probabilities corresponding to each channel. If None, equal probabilities are assumed.
        """
        self.channels = []
        if probs is None:
            probs = [1/len(channels)] * len(channels)  
        assert len(channels) == len(probs), "Length of channels and probs must match." 
        for i, channel in enumerate(channels):
            self.channels.append((probs[i], channel))

if __name__ == "__main__":
    form = 'pauli'
    H = (['ZZ', 'ZI', 'IZ','XI','IX',], [-1.0, -0.5, -0.5,0.25,0.25])
    L = [np.array([[1, 0,0, 0], [0, np.sqrt(1/3), -np.sqrt(2/3), 0], [0, np.sqrt(2/3), np.sqrt(1/3), 0], [0,0,0,1]])]

    QIMF_amp = Lindbladian(H, L)
    for L in QIMF_amp.L_list:
        for L_mat in L.mat_list:
            print(L_mat.mat, L_mat.coeff)
        

