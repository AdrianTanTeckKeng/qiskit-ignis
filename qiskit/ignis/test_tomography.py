# Needed for functions
import numpy as np
import time

# Import Qiskit classes
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
#sys.path.append('/Users/atanteck/Desktop/project/qiskit-ignis/qiskit/ignis')
from verification import tomography as tomo
from verification.tomography.fitters.pure_state_mle_fit import pure_state_mle_fit
from mitigation import measurement as mc

import matplotlib.pyplot as plt 
	# Create a state preparation circuit
f_list = []
angle_list = []
N = 1

def rand_angles():
	return tuple(2 * np.pi * np.random.random(3) - np.pi)

for i in range(0,N):
	q2 = QuantumRegister(2)
	bell = QuantumCircuit(q2)
	'''
	for j in range(2):
		bell.u3(*rand_angles(),q2[j])
	'''
	#bell.ry(np.pi/2,q2[0])
	#bell.rx(np.pi/2,q2[1])
	#bell.h(q2[0])
	#bell.cx(q2[0], q2[1])
	bell.h(q2[0])
	#bell.ry(np.pi/2,q2[1])
	job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
	psi_bell = job.result().get_statevector(bell)

	# Generate circuits and run on simulator
	t = time.time()
	qst_bell = tomo.state_tomography_circuits(bell, q2,special_labels=False)
	job = qiskit.execute(qst_bell, Aer.get_backend('qasm_simulator'), shots=5000)
	print('Time taken:', time.time() - t)

	# Extract tomography data so that countns are indexed by measurement configuration
	# Note that the None labels are because this is state tomography instead of process tomography
	# Process tomography would have the preparation state labels there

	tomo_counts_bell = tomo.tomography_data(job.result(), qst_bell,efficient=True)
	#tomo_counts_bell.update({('X','I'): tomo_counts_bell[('X','X')]})
	#tomo_counts_bell.pop(('X','X'),None)
	for key in tomo_counts_bell:
		print(key,":",tomo_counts_bell[key])
	# Generate fitter data and reconstruct density matrix
	probs_bell, basis_matrix_bell, weights_bell = tomo.fitter_data(tomo_counts_bell)
	rho_bell = pure_state_mle_fit(probs_bell, basis_matrix_bell, weights_bell)
	#rho_bell = rho_bell[0][0]
	print(rho_bell[0][0]/rho_bell[0][1])
	#print(type(rho_bell[0][0]))
	ddddd
	F_bell = state_fidelity(psi_bell, rho_bell)
	f_list.append(F_bell)
	print('Circuit: ',i,' ','Fit Fidelity =', F_bell)
'''
plt.plot(f_list)
plt.grid()
plt.xlabel("Trial")
plt.xlim([0,N])
plt.ylabel("fidelity")
plt.show()
'''