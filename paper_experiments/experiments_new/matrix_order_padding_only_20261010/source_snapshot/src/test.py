from qiskit import QuantumCircuit, transpile

custom_circuit = QuantumCircuit(2, name='bell')
custom_circuit.h(0)
custom_circuit.cx(0, 1)

custom_gate = custom_circuit.to_gate()

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.append(custom_gate, [0,1])
circuit.cx(1, 2)
circuit.draw()


basis_gates = ['bell', 'u3', 'cx']

qc_trans = transpile(circuit, basis_gates=basis_gates)
qc_trans.draw()