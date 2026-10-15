###################
#                                #
# State preparation subroutines. #
#                                #
#
###################

import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import  XGate
from qiskit_aer import AerSimulator

from Outdated.qsvt import  phase_generator
i
def add_ancilla_control(circuit: QuantumCircuit):
    ancilla_reg = QuantumRegister(1, name='ancilla')

    new_qc = QuantumCircuit(ancilla_reg, *circuit.qregs)
    
    control_qubit = new_qc.qubits[0]

    for inst in circuit.data:
        orig_gate = inst.operation
        orig_qubits = inst.qubits
        if orig_gate.name == 'barrier':
            
            new_qubits = [control_qubit] + list(orig_qubits)
            new_qc.append(orig_gate, new_qubits, [])
            continue
        try:
            controlled_gate = orig_gate.control(1)
            new_qubits = [control_qubit] + list(orig_qubits)
            
            new_qc.append(controlled_gate, new_qubits, [])
        except Exception as e: 
            print(f"Could not control gate {orig_gate}: {e}")
            raise e
        
    return new_qc
## Prepare a uniform superposition over M states using n qubits.
def prep_uniform_superposition(N: int):
    """
    Prepare a uniform superposition over M states using n qubits.
    
    Args:
        M (int): Number of states to prepare the superposition over.
        n (int): Number of qubits to use.
        
    Returns:
        QuantumCircuit: A quantum circuit that prepares the desired state.
    """
    
    # If N == 1, return the |0> state (base case)
    # if N == 1:
    #     qc = QuantumCircuit(1)
    #     return qc, 1
    # Calculate the number of qubits needed to represent N states

    num_qubits = int(np.ceil(np.log2(N + 1)))
    first_half = 2**(num_qubits - 1)
    second_half = N - first_half
    # Prepare uniform superposition over 2^num_qubits_needed states
    # If not the topmost layer, do not apply
    rot_angle = np.arccos(np.sqrt(first_half/N))
    qc = QuantumCircuit(num_qubits)
    if rot_angle != 0:
        qc.ry(2 * rot_angle, 0)
    # 0-controlled hadamards to the first half
    qc.x(0)
    for i in range(1, num_qubits):
        qc.ch(0, i)
    # qc.barrier()
    qc.x(0)

    if second_half <= 1 :
        return qc, num_qubits
    
    qc_low_level, num_qubits_low_level = prep_uniform_superposition(second_half)
    # Add control to the low-level circuit conditioned on the 1st qubit being |1>
    qc_low_level = add_ancilla_control(qc_low_level)
    # qc_low_level = qc_low_level.to_gate(label = "low_level_qc")

    mapping_list = [0] + list(range(num_qubits - num_qubits_low_level, num_qubits))
    qc = qc.compose(qc_low_level, qubits=mapping_list)
    # qc.append(qc_low_level, mapping_list)
    # qc = qc.decompose(gates_to_decompose=["low_level_qc"])
    # Prepare rotation 
    return qc, num_qubits

def prepare_sine_state(n: int):
    """
    A (1,1,0)-block encoding to prepare the sine coeffient superposition states for -2^n to 2^n-1. """

    qc = QuantumCircuit(n + 1)
    qc.h(0)
    for i in range(1, n + 1):
        qc.cx(0, i)
    # qc.barrier()
    for i in range(1, n + 1):
        angle = 1.0 / np.pow(2, n - i) if i < n else -1
        qc.rz(angle, i)
    # qc.barrier()
    for i in range(1, n + 1):
        qc.cx(0, i)
    # qc.barrier()
    qc.rz(1.0 / np.pow(2, n - 1), 0)
    qc.h(0)
    qc.y(0)
    return qc

def prepare_func_state(func, n: int, degree: int ):
    """
    A block-encoding method to prepare the superposition state
    with amplitudes given by \tilde{func}(x), the approximation of func(x). 
    Args:
        func (callable): The function to approximate.
        n (int): Number of qubits to use.
        degree (int): Degree of the approximating polynomial, also the length of the sequence. 
    Returns:
        QuantumCircuit: A quantum circuit that prepares the desired state.
    Modules:
    Step 1. Prepare a sine state using prepare_sine_state.
    Step 2. Apply QSVT to approximate func using a fixed-parity polynomial h(sin(x)).
    """
    
    ### Use n qubits to represent 2^n states from -2^{n-1} to 2^{n-1} - 1

    phase_coeffs, targ_amp = phase_generator(func, np.sqrt(1/2), 2**n,  degree) ## Return QSVT angles and success probability from QSP phase generators
    qc = QuantumCircuit(n + 2) ## An additional ancilla q for qsvt
    qc_sin = prepare_sine_state(n) ## circuit of n + 1 qubits
    qc_sin_inv = qc_sin.inverse()
    # Initialize the ancilla 
    qc.h(0)
    ocx = XGate().control(1, ctrl_state='0') # 0-controlled X gate
    phase_coeffs = phase_coeffs[1:-1] # Remove the first and last constant phases
    # Apply the QSVT unit
    for i in range(degree - 1):
        if i % 2 == 0:
            qc = qc.compose(qc_sin, qubits = range(1, n + 2)) #type: ignore
        else:
            qc = qc.compose(qc_sin_inv, qubits = range(1, n + 2)) #type: ignore
        qc.append(ocx, [1, 0])
        qc.rz(2 * phase_coeffs[-i], 0)
        qc.append(ocx, [1, 0])

    return qc, targ_amp
    
    ## Apply QSVT according to the degree
     # for i in range(degree):
def MCZ0gate(n: int):
    """
    Create a multi-controlled Z gate with n control qubits of value 0.
    
    Args:
        n (int): Number of control qubits.
        
    Returns:
        QuantumCircuit: A quantum circuit representing the MCZ gate.
    """
    qc = QuantumCircuit(n + 1)
    qc.h(0)
    for i in range(n):
        qc.x(i + 1)
    # qc.barrier()
    qc.mcx(list(range(1, n + 1)), 0)
    for i in range(n):
        qc.x(i + 1)
    qc.h(0)
    # qc.barrier()
    return qc

def exact_amp_amplification(circuit: QuantumCircuit, target_amplitude: float):
    """
    Apply exact amplitude amplification to boost the success probability of preparing a target state.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit that prepares the initial state.
        target_amplitude (float): The amplitude of the target state in the initial state.
        
    Returns:
        QuantumCircuit: A quantum circuit that applies amplitude amplification.
    """
    k = int(np.pi / (4 * np.arcsin(target_amplitude)) + 0.5)
    theta = np.pi / (4 * k + 2)
    omega = 2 * np.arccos(np.sin(theta)/ target_amplitude)
    Uf = circuit.to_gate(label="Uf")
    numq_Uf = circuit.num_qubits ## numq_Uf = n + 2 for n = log_2(N)\
    qreg = QuantumRegister(numq_Uf + 1, name='q')
    creg = ClassicalRegister(3, name = 'c')
    qc = QuantumCircuit(qreg, creg)
    ## Base implementation
    qc.h(range(3, numq_Uf + 1))
    # qc.barrier()
    qc.append(Uf, range(1, numq_Uf + 1))
    qc.ry(2 * omega, 0)
    
    ### Amplification
    for i in range(k):
        qc.compose(MCZ0gate(2), range(0, 3), inplace = True)
        qc.append(Uf.inverse(), range(1, numq_Uf + 1))
        qc.ry(-2 * omega, 0)
        qc.h(range(3, numq_Uf + 1))
        # qc.barrier()
        qc.compose(MCZ0gate(numq_Uf), range(0, numq_Uf + 1), inplace=True)
        qc.h(range(3, numq_Uf + 1))
        # qc.barrier()
        qc.append(Uf, range(1, numq_Uf + 1))
        qc.ry(2 * omega, 0)

    for i in range(0, 3):
        qc.measure(i, creg[i])
    return qc

def simulate_circuit(circuit, shots: int):
    """
    Simulate a quantum circuit and return the measurement counts.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to simulate.
        shots (int): Number of shots for the simulation.
    """

    simulator = AerSimulator()
    init_state = Statevector.from_label('0' * circuit.num_qubits)
    ## add measurements 
    circuit.measure_all()
    circuit = transpile(circuit, simulator)
    result = simulator.run(circuit, shots = shots).result()
    counts = result.get_counts()
    return counts

def transpile_to_gates(circuit: QuantumCircuit, basis_gates: list, opt_level: int, backend = None):
    """
    Transpile a quantum circuit to a specified set of basis gates.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to transpile.
        basis_gates (list): List of basis gate names to transpile to.
        
    Returns:
        QuantumCircuit: The transpiled quantum circuit.
    """
    if backend is not None: 
        transpiled_qc = transpile(circuit, backend = backend, 
                                  basis_gates = basis_gates, optimization_level= opt_level)
    else:
        transpiled_qc = transpile(circuit, basis_gates = basis_gates, optimization_level= opt_level)
    return transpiled_qc
if __name__ == "__main__":

    basis_gates = ['rx','ry','cx','h','s','sdg','x','y','z']

    func  = lambda x: np.pow(2/np.pi, 1/4) * np.exp(-x**2)
    qc_sin = prepare_sine_state(5)

    # print(qc_sin.draw())
    qc_sin_t = transpile_to_gates(qc_sin, basis_gates, opt_level=3)
    qc, targ_amp = prepare_func_state(func, 4, degree = 8)
    # print(qc.draw())
    qc_t = transpile_to_gates(qc, basis_gates, opt_level=3)

    
    # qc = QuantumCircuit(4)
    # qc.h(range(4))
    # qc.cx(0, 3)
    # qc.ry(np.pi/4, 3)
    # targ_amp = 0.4
    qc_amp = exact_amp_amplification(qc, targ_amp)
    print(qc_amp.draw())

    exit(0)
    qc, num_qubits = prep_uniform_superposition(22)
    print(qc.draw())
    counts = simulate_circuit(qc, shots=8192)
    print(counts)

