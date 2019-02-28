import numpy as np

import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import qiskit.ignis.verification.tomography as tomo

from qiskit.ignis.verification.tomography.fitters.pure_state_mle_fit import pure_state_mle_fit, pure_state_mle_fit_density_matrix

from scipy.linalg import sqrtm

qr1 = QuantumRegister(2)
bell1 = QuantumCircuit(qr1)
bell1.h(qr1[0])
bell1.cx(qr1[0], qr1[1])

job = qiskit.execute(bell1, Aer.get_backend('statevector_simulator'))
psi_bell1_real = job.result().get_statevector(bell1)

qst_bell1 = tomo.state_tomography_circuits(bell1, qr1)
job = qiskit.execute(qst_bell1, Aer.get_backend('qasm_simulator'), shots=5000)

tomo_counts_bell1 = tomo.tomography_data(job.result(), qst_bell1, efficient = True)

probs_bell1, basis_matrix_bell, _ = tomo.fitter_data(tomo_counts_bell1)



qr2 = QuantumRegister(2)
bell2 = QuantumCircuit(qr2)
bell2.x(qr2[0])
bell2.h(qr2[0])
bell2.cx(qr2[0], qr2[1])

job = qiskit.execute(bell2, Aer.get_backend('statevector_simulator'))
psi_bell2_real = job.result().get_statevector(bell2)

rho_bell_real = [[0 for i in range(len(psi_bell1_real))] for j in range(len(psi_bell1_real))]
for i in range(len(psi_bell1_real)):
    for j in range(len(psi_bell2_real)):
        rho_bell_real[i][j] = 0.9 * psi_bell1_real[i].conj()*psi_bell1_real[j] + 0.1 * psi_bell2_real[i].conj()*psi_bell2_real[j]

qst_bell2 = tomo.state_tomography_circuits(bell2, qr2)
job = qiskit.execute(qst_bell2, Aer.get_backend('qasm_simulator'), shots=5000)

tomo_counts_bell2 = tomo.tomography_data(job.result(), qst_bell2, efficient = True)

probs_bell2, _, _ = tomo.fitter_data(tomo_counts_bell2)

probs_total = [0.9*probs_bell1[i]+0.1*probs_bell2[i] for i in range(len(probs_bell1))]

psi_bell, diff = pure_state_mle_fit(probs_total, basis_matrix_bell)
pure_mle_fidelity = 0
for i in range(len(psi_bell)):
    for j in range(len(psi_bell)):
        pure_mle_fidelity += psi_bell[i].conj()*psi_bell[j]*rho_bell_real[i][j]
#my_fidel = sum([psi_bell[i]*psi_bell_real[i] for i in range(len(psi_bell))])
print("Psi: {}".format(psi_bell))
print("Diff: {}".format(diff))
#rho = pure_state_mle_fit_density_matrix(probs_bell, basis_matrix_bell)
#print("rho: {}".format(rho))

rho_full_mle = tomo.state_mle_fit(probs_total, basis_matrix_bell)
#F_bell = state_fidelity(psi_bell, rho_full_mle)

full_mle_fidelity = np.trace(sqrtm(sqrtm(rho_full_mle).dot(rho_bell_real).dot(sqrtm(rho_full_mle))))
print("Full fidelity: ", full_mle_fidelity)
print("Pure fidelity: ", pure_mle_fidelity)

#print('Fit Fidelity =', F_bell)
#print('my fidelity =', my_fidel)
print(np.linalg.eig(rho_full_mle))
