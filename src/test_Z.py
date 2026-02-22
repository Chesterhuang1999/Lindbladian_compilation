from qiskit import QuantumCircuit, QuantumRegister



from qiskit.quantum_info import Pauli

# 定义 Pauli 对象
p = Pauli('IIXXXZIYY')

# 转换为字符串并计算
label = p.to_label()
non_i_count = len(label) - label.count('I')

print(f"非I项数: {non_i_count}") 
# 输出: 非I项数: 6
