import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PhaseGate, ZGate, RZGate
from qiskit.quantum_info import Operator, Statevector
import matplotlib.pyplot as plt
from series_expansion import block_encoding_matrixsum
from subroutine import phase_generator
from scipy.linalg import expm, sinm, cosm, norm
from channel_IR import Lindbladian

def qsvt_circuit(H_encode, phi_angles, system_qubits, ancilla_qubits):
    """
    构建标准的QSVT电路
    
    参数:
    H_encode: 哈密顿量H的block-encoding电路 (U = [[H, ·], [·, ·]])
    phi_angles: 相位角度序列 [φ_0, φ_1, ..., φ_d]
    system_qubits: 系统量子比特的QuantumRegister或列表
    ancilla_qubits: block-encoding辅助量子比特的QuantumRegister或列表
    
    返回:
    qc: QSVT电路
    """
    # 确保输入格式正确
    if not isinstance(system_qubits, list):
        system_qubits = list(system_qubits)
    if not isinstance(ancilla_qubits, list):
        ancilla_qubits = list(ancilla_qubits)
    
    # 创建电路
    qc = QuantumCircuit()
    
    # 添加所有量子比特
    # for qubit in system_qubits + ancilla_qubits:
    #     qc.add_q(qubit)
    qc.add_register(system_qubits)
    qc.add_register(ancilla_qubits)
    
    # 获取所有量子比特的索引
    all_qubits = list(range(len(system_qubits) + len(ancilla_qubits)))
    ancilla_indices = list(range(len(system_qubits), len(system_qubits) + len(ancilla_qubits)))
    
    # QSVT算法步骤:
    # 1. 初始相位旋转在第一个ancilla上
    qc.rz(2 * phi_angles[0], ancilla_indices[0])
    
    # 2. 交替应用U和U†，中间穿插相位旋转
    d = len(phi_angles) - 1  # 多项式次数
    
   

    for j in range(1, d + 1):
        # 应用block-encoding U 或其共轭转置
        if j % 2 == 1:
            # 奇数次: 应用U
            # 需要将H_encode添加到正确的位置
            qc.append(H_encode.to_instruction(), all_qubits)
        else:
            # 偶数次: 应用U† (U的逆)
            qc.append(H_encode.inverse().to_instruction(), all_qubits)
        
        # 在第一个ancilla上应用相位旋转
        if j < d:  # 最后一次循环后不需要相位旋转
            qc.rz(2 * phi_angles[j], ancilla_indices[0])
    qc.rz(2 * phi_angles[d], ancilla_indices[0])
    return qc

def hamiltonian_simulation_qsvt(H_encode, time, system_qubits, ancilla_qubits, 
                                approximation_degree, norm_factor):
    """
    使用QSVT进行哈密顿量模拟
    
    参数:
    H_encode: block-encoding电路
    time: 演化时间
    system_qubits: 系统量子比特
    ancilla_qubits: 辅助量子比特
    approximation_degree: 多项式近似阶数
    norm_factor: H的范数因子(α, 使得 ||H|| ≤ α)
    
    返回:
    qc: 完整的哈密顿量模拟电路
    phi_cos: 用于cos(Ht)的相位序列
    phi_sin: 用于sin(Ht)的相位序列
    """
    # 计算相位序列
    # phi_cos = compute_qsvt_phases_cos(time, approximation_degree, norm_factor)
    # phi_sin = compute_qsvt_phases_sin(time, approximation_degree, norm_factor)
    cos_func = lambda x: np.cos(time * norm_factor * x)
    sin_func = lambda x: np.sin(time * norm_factor * x)
    
    N = 1000
    phi_cos = phase_generator(cos_func, N, approximation_degree)
    phi_sin= phase_generator(sin_func, N, approximation_degree - 1)


    # print(f"cos相位: {phi_cos}")
    # print(f"sin相位: {phi_sin}")
    
    # 创建电路
    qc = QuantumCircuit()
    
    # 添加系统量子比特
    system_reg = QuantumRegister(len(system_qubits), 'system')
    ancilla_reg = QuantumRegister(len(ancilla_qubits), 'ancilla')
    qc.add_register(system_reg)
    qc.add_register(ancilla_reg)
    
    # 额外ancilla用于线性组合
    flag_qubit = QuantumRegister(1, 'flag')
    qc.add_register(flag_qubit)
    
    # 准备flag qubit为|+⟩态
    qc.h(flag_qubit[0])
    

    # 当flag qubit为|0⟩时执行exp(-i \phi)
    qc = add_controlled_qsvt(qc, H_encode, phi_cos, flag_qubit[0], 
                            system_reg, ancilla_reg, control_state=0)
    # qc = qsvt_circuit(H_encode, phi_cos, system_reg, ancilla_reg)
    
    # 当flag qubit为|1⟩时执行exp(i\phi)
    # qc.z(flag_qubit[0])  # 添加-i相位
    # qc = add_controlled_qsvt(qc, H_encode, phi_sin, flag_qubit[0],
    #                         system_reg, ancilla_reg, control_state=1)
    # qc.z(flag_qubit[0])  # 抵消之前的Z门（如果需要）
    
    # 最后应用Hadamard恢复
    qc.h(flag_qubit[0])
    
    return qc, phi_cos, phi_sin
def PCPhase(qc, target, control, phi_angle):
    qc.x(control)
    qc.cx(control, target)
    qc.rz(2 * phi_angle, target)
    
    qc.cx(control, target)
    qc.x(control)
def add_controlled_qsvt(circuit, H_encode, phi_angles, control_qubit,
                       system_qubits, ancilla_qubits, control_state=0):
    """
    添加受控的QSVT电路
    """
    qc = circuit.copy()
    
    # 获取所有量子比特
    all_qubits = list(system_qubits) + list(ancilla_qubits)
    
    # 获取索引
    ancilla_index = len(system_qubits)  # 第一个ancilla的索引
    
    # 初始相位旋转（受控）
    if control_state == 0:
        # 控制为0时激活
        PCPhase(qc, control_qubit, ancilla_qubits[0], phi_angles[0])
    else:
        # 控制为1时激活：使用X门反转控制
        qc.x(control_qubit)
        qc.crz(phi_angles[0], control_qubit, ancilla_qubits[0])
        qc.x(control_qubit)
    
    d = len(phi_angles) - 1
    
    for j in range(1, d + 1):
        # 应用受控的U或U†
        if j % 2 == 1:
            # 奇数次: 应用U
            # controlled_u = H_encode.control(num_ctrl_qubits=1, ctrl_state=control_state)
            # qc.append(controlled_u, [control_qubit] + all_qubits)
            qc.compose(H_encode, all_qubits, inplace=True)
        else:
            # 偶数次: 应用U†
            controlled_u_dag = H_encode.inverse().control(num_ctrl_qubits=1, ctrl_state=control_state)
            # qc.append(controlled_u_dag, [control_qubit] + all_qubits)
            qc.compose(H_encode.inverse(), all_qubits, inplace=True)
        
        # 受控的相位旋转
        if j < d + 1:
            if control_state == 0:
                # qc.crz(phi_angles[j], control_qubit, ancilla_qubits[0])
                PCPhase(qc, control_qubit, ancilla_qubits[0], phi_angles[j])

            else:
                qc.x(control_qubit)
                qc.crz(phi_angles[j], control_qubit, ancilla_qubits[0])
                qc.x(control_qubit)
    
    return qc

def compute_qsvt_phases_cos(time, degree, norm_factor=1.0):
    """
    计算cos(Ht)近似的QSVT相位序列
    
    注意：这是一个简化的示例。实际应用中，相位计算更复杂，
    需要根据具体多项式近似和奇偶性进行调整。
    """
    # 简化的相位计算：实际应用需要更精确的方法
    # 这里使用一个简单的线性分布作为示例
    phi = np.zeros(degree + 1)
    
    # 实际相位计算应考虑：
    # 1. 目标函数f(x) = cos(αtx) 其中x = H/α
    # 2. 使用Chebyshev或其他多项式近似
    # 3. 通过QSVT相位查找算法计算相位
    
    # 示例：简单的线性相位
    for k in range(degree + 1):
        phi[k] = (-1)**k * time * norm_factor / (degree + 1)
    
    return phi

def compute_qsvt_phases_sin(time, degree, norm_factor=1.0):
    """
    计算sin(Ht)近似的QSVT相位序列
    """
    # 简化的相位计算
    phi = np.zeros(degree + 1)
    
    # 示例：简单的线性相位
    for k in range(degree + 1):
        phi[k] = (-1)**k * time * norm_factor / (degree + 1) + np.pi/2
    
    return phi

# 测试示例
def test_qsvt_hamiltonian_simulation():
    """
    测试QSVT哈密顿量模拟
    """
    print("=== 测试QSVT哈密顿量模拟 ===")
    
    # 创建一个简单的block-encoding电路示例
    # 假设H = X (Pauli-X门)，其block-encoding为CNOT门
    n_system = 2
    n_ancilla = 1
    

    # 演化时间
    time = 1.0
    H = [('XI', -1), ('IX', -1)]
    L_list = []
    TFIM_lind = Lindbladian(H, L_list)
    eff_H = TFIM_lind.H.eff_op()
    H_encode = block_encoding_matrixsum(TFIM_lind.H)
    ini_state = Statevector.from_label('0+')
    H_evo = Operator(expm(-1j * time * eff_H ))
    H_cos = Operator(cosm(time * eff_H))
    H_sin = Operator(sinm(time * eff_H))
    
    print(f"演化结果的baseline: {ini_state.evolve(H_cos)}")
    # print(f"Block-encoding电路:\n{H_encode.draw()}")
    
    
    # 量子寄存器
    system_q = QuantumRegister(n_system, 'system')
    ancilla_q = QuantumRegister(n_ancilla, 'ancilla')
    
    
    # 构建QSVT哈密顿量模拟电路
    qsvt_circuit, phi_cos, phi_sin = hamiltonian_simulation_qsvt(
        H_encode, time, system_q, ancilla_q, 
        approximation_degree=4, norm_factor=TFIM_lind.H.pauli_norm()
    )
    
    # print(f"\nQSVT电路深度: {qsvt_circuit.depth()}")
    # print(f"量子比特数: {qsvt_circuit.num_qubits}")
    # print(f"门数量: {qsvt_circuit.size()}")
    
    # 绘制电路（仅显示前部分）
    print("\n电路结构（简化）:")
    print(qsvt_circuit.draw())
    # 验证电路功能
    print("\n=== 电路验证 ===")
    
    # 准备测试态 |+⟩⊗|0⟩
    test_state = Statevector.from_label('0' * (1 + n_ancilla) + '0+')
    
    # 应用电路
    final_state = test_state.evolve(qsvt_circuit).data
    # final_state = [final_state[i] * np.exp(-1j * np.pi /4) for i in range(len(final_state))]

    # print(f"初始态: |+⟩^{n_system} |0⟩^{n_ancilla}")
    print(f"final state(系统部分: {final_state[:2**n_system]})")
    
    return qsvt_circuit, phi_cos, phi_sin

# 运行测试
if __name__ == "__main__":
    # qc = QuantumCircuit(2)
    # qc.x(1)
    # qc.cx(1, 0)
    # qc.rz(np.pi /2 , 0)
    # qc.cx(1, 0)
    # qc.x(1)
    # print(RZGate(-np.pi /2).to_matrix())
    # print(Operator(qc.to_gate()))
    # Z = Operator(ZGate())
    # rz = Operator(expm(1j * Z.tensor(Z) * np.pi / 4))
    # iden = Operator(np.eye(2))
    # print(rz)

    test_qsvt_hamiltonian_simulation()
