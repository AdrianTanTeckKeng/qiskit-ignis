import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import qiskit.ignis.verification.tomography as tomo

from qiskit.ignis.verification.tomography.fitters.pure_state_mle_fit import pure_state_mle_fit, pure_state_mle_fit_density_matrix


q2 = QuantumRegister(2)
bell = QuantumCircuit(q2)
bell.h(q2[0])
bell.cx(q2[0], q2[1])

qst_bell = tomo.state_tomography_circuits(bell, q2)
job = qiskit.execute(qst_bell, Aer.get_backend('qasm_simulator'), shots=5000)

tomo_counts_bell = tomo.tomography_data(job.result(), qst_bell, efficient = True)

probs_bell, basis_matrix_bell, weights_bell = tomo.fitter_data(tomo_counts_bell)
#psi_bell, diff = pure_state_mle_fit(probs_bell, basis_matrix_bell)
#print("Psi: {}".format(psi_bell))
#print("Diff: {}".format(diff)
rho = pure_state_mle_fit_density_matrix(probs_bell, basis_matrix_bell)
print("rho: {}".format(rho))
