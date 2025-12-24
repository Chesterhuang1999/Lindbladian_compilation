######################
# Implementation of time-controlled Hamiltonian simulation # 
#          using Trotter-Suzuki decomposition              #
#              
#######################

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZZGate, RXXGate, RYYGate
import numpy as np

