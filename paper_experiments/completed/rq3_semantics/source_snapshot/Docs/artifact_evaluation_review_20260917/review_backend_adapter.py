"""Review candidate: fixed residual and explicit [work, system] block interface."""
from qiskit import QuantumCircuit, QuantumRegister
import block_encoding_new
from mobius_residual_fixed import mobius_invert_modes_with_phases

class PackedBlockEncoding(block_encoding_new.BlockEncoding):
    def circuit(self,opt=None):
        qc=super().circuit(opt=opt)
        systems=[q for reg in qc.qregs if reg.name=='sys' for q in reg]
        if len(systems)!=self.sys_size:
            raise ValueError(f'Expected {self.sys_size} system qubits; found {len(systems)}')
        work=[q for q in qc.qubits if q not in systems]
        desired=work+systems
        if desired==list(qc.qubits):return qc
        regs=[]
        if work:regs.append(QuantumRegister(len(work),'work'))
        regs.append(QuantumRegister(len(systems),'sys'))
        result=QuantumCircuit(*regs,name=qc.name)
        # compose preserves the original global phase as well as its operations.
        result.compose(qc,qubits=[desired.index(q) for q in qc.qubits],inplace=True)
        return result

def install():
    block_encoding_new.mobius_invert_modes_with_phases=mobius_invert_modes_with_phases
    import run_cp2_random_ablation as random_runner
    import run_cp2_hrw20_ablation as common_runner
    random_runner.NewBlockEncoding=PackedBlockEncoding
    common_runner.NewBlockEncoding=PackedBlockEncoding
