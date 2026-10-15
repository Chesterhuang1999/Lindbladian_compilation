from block_encoding import *
from channel_IR import * 
from qiskit.circuit.library import StatePreparation, XGate, CCXGate, CXGate
from qiskit import QuantumCircuit, QuantumRegister, transpile

def optimize_channel_LCU_syntax(circuits: list):
    
    select_size = int(np.ceil(np.log2(len(circuits))))
    sys_size = max(circ.num_qubits for circ in circuits)
    mccount = 0
    tcount = 0
    cxcount = 0
    t_counts_per_ccx = 4
    cx_counts_per_ccx = 4  
    opt_circuit = QuantumCircuit(1 + 2 * select_size + sys_size)
    opt_circuit.x(0)
    sel_regs = [2 * j + 1 for j in range(select_size)]
    anc_regs = [2 * j + 2 for j in range(select_size)]
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
    maxctrl_value = bin(len(circuits) - 1)[2:].zfill(select_size)
    def find_bit_to_remove(max_val, cur_val):
        candidate_bits = []
        for j in range(select_size):
            if max_val[j] != cur_val[j]:
                return candidate_bits
            elif max_val[j] == '0' and cur_val[j] == '0':
                candidate_bits.append(j)
            else:
                continue
        return candidate_bits
    for i, circ in enumerate(circuits):
        numq = circ.num_qubits
        control_values = bin(i)[2:].zfill(select_size)
        index_remove = find_bit_to_remove(maxctrl_value, control_values)
        print(index_remove)
        if i == 0:
            cval_next = bin(1)[2:].zfill(select_size)   
            for j in range(select_size):
                apply_left_enc(j)
                mccount += 1
                tcount += t_counts_per_ccx
                cxcount += cx_counts_per_ccx
            opt_circuit.append(circ.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * select_size,2 * select_size + 1 +  numq)))
            opt_circuit.cx(anc_regs[select_size - 2], anc_regs[select_size - 1])
            cxcount += 1
        elif i == len(circuits) - 1:
            cval_prev = bin(i - 1)[2:].zfill(select_size)
            diff_prev = next(j for j in range(select_size) if cval_prev[j] != control_values[j])
            if diff_prev != select_size - 1:
                for j in range(diff_prev + 1, select_size):
                    
                    apply_left_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
            opt_circuit.append(circ.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * select_size,2 * select_size + 1 + numq)))
            for j in range(select_size - 1, -1, -1):
                apply_right_enc(j)
                mccount += 1
                tcount += t_counts_per_ccx
                cxcount += cx_counts_per_ccx
        else:
            cval_prev, cval_next = bin(i - 1)[2:].zfill(select_size), bin(i + 1)[2:].zfill(select_size)
            ## Find the first bit that differs
            diff_prev = next(j for j in range(select_size) if cval_prev[j] != control_values[j])
            ## Apply left encodings from diff_prev to the end
            if diff_prev != select_size - 1:
                for j in range(diff_prev + 1, select_size):
                    apply_left_enc(j)
                    mccount += 1
                    tcount += t_counts_per_ccx
                    cxcount += cx_counts_per_ccx
            ## Apply the controlled circuit
            opt_circuit.append(circ.to_gate().control(num_ctrl_qubits = 1, ctrl_state = '1'), list(range(2 * select_size,2 * select_size + 1 + numq)))
            diff_next = next(j for j in range(select_size) if cval_next[j] != control_values[j])
            ## Apply right encodings from diff_next to the end

            if diff_next != select_size - 1:
                for j in range(select_size - 1, diff_next, -1):
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
    return opt_circuit, mccount, tcount, cxcount

def optimize_full_Paulis(n: int):
    """
    The optimized circuit for implementing the full Pauli group on n qubits, which consists of 4^n elements.
    """
    w = int(np.log2(4**n))

    qc = QuantumCircuit(w + n)
    for i in range(w):
        if i < n:
            qc.cx(i, i+ w)
        else:
            qc.cz(i, i - n + w)
    
    for i in range(n):
        qc.cp(np.pi / 2, i, i + n)
    return qc
def detailed_gate_count(circuit, custom_gates=None):
    """
    更详细的门计数函数，支持自定义门统计
    
    参数:
        circuit (QuantumCircuit): 要分析的量子电路
        custom_gates (list): 额外要统计的门名称列表
    
    返回:
        dict: 详细的门计数统计
    """
    if custom_gates is None:
        custom_gates = []
    
    # 基础门集合
    base_gates = ['cx', 'h', 's', 't', 'tdg', 'x', 'y', 'z', 'p', 'u']
    all_gates_to_count = list(set(base_gates + custom_gates))
    
    # 使用defaultdict自动处理未见过的门
    gate_counts = defaultdict(int)
    
    # 统计总门数和总量子比特数
    total_gates = 0
    total_qubits = circuit.num_qubits
    
    # 遍历电路数据
    for instruction in circuit.data:
        gate_name = instruction[0].name
        total_gates += 1
        
        # 统计指定的门
        if gate_name in all_gates_to_count or any(gate in gate_name for gate in all_gates_to_count):
            gate_counts[gate_name] += 1
    
    # 计算其他统计信息
    stats = {
        
        'total_gates': total_gates,
        'total_qubits': total_qubits,
        'circuit_depth': circuit.depth(),
        'specific_gates': dict(gate_counts),
        'gate_density': total_gates / total_qubits if total_qubits > 0 else 0
    }
    
    return stats

if __name__ == "__main__":
    # cand = [3, 4, 6, 8]
    # for j in cand:
    #     n = j
    #     w = int(np.log2(4**n))
    #     qc = optimize_full_Paulis(n)
    # # ctrl_state_ini = Statevector.from_label('01100110')
    # # ctrl_state_tot = Statevector.from_label('0' * n).tensor(ctrl_state_ini)
    #     qc_clift = transpile(qc, basis_gates = ['cx', 'h', 's', 't'])
    #     print(detailed_gate_count(qc_clift))
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc = transpile(qc, basis_gates = ['cx', 'h', 's', 't', 'tdg', 'sdg'])
    print(qc.draw())
    print(detailed_gate_count(qc))
    circ_set = []
    for j in range(11):
        circ = QuantumCircuit(2)
        circ.h(0)
        circ_set.append(circ)
    circuit, mccount, tcount, cxcount = optimize_channel_LCU_syntax(circ_set)
    print(circuit.draw())
    print(f"mccount: {mccount}, tcount: {tcount}, cxcount: {cxcount}")  
                
