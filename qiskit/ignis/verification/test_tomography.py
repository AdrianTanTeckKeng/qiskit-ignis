# Needed for functions
import numpy as np
import time

# Import Qiskit classes
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import tomography as tomo

# Create a state preparation circuit
q2 = QuantumRegister(2)
bell = QuantumCircuit(q2)
bell.h(q2[0])
bell.cx(q2[0], q2[1])
print(bell.qasm())

job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
psi_bell = job.result().get_statevector(bell)
print(psi_bell)

# Generate circuits and run on simulator
t = time.time()
qst_bell = tomo.state_tomography_circuits(bell, q2,special_labels=False)
job = qiskit.execute(qst_bell, Aer.get_backend('qasm_simulator'), shots=5000)
print('Time taken:', time.time() - t)

# Extract tomography data so that countns are indexed by measurement configuration
# Note that the None labels are because this is state tomography instead of process tomography
# Process tomography would have the preparation state labels there

tomo_counts_bell = tomo.tomography_data(job.result(), qst_bell)
tomo_counts_bell.pop(('X','Y'),None)
tomo_counts_bell.pop(('X','Z'),None)
#tomo_counts_bell.update({('X','I'): tomo_counts_bell[('X','X')]})
#tomo_counts_bell.pop(('X','X'),None)
for key in tomo_counts_bell:
	print(key,":",tomo_counts_bell[key])
# Generate fitter data and reconstruct density matrix
probs_bell, basis_matrix_bell, weights_bell = tomo.fitter_data(tomo_counts_bell)
print('-------------------------------------------------------------------------------')
print(np.shape(probs_bell))
print(np.shape(basis_matrix_bell))
print(np.shape(weights_bell))
dddd
rho_bell = tomo.state_cvx_fit(probs_bell, basis_matrix_bell, weights_bell)
F_bell = state_fidelity(psi_bell, rho_bell)
print('Fit Fidelity =', F_bell)