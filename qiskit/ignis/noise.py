# Needed for functions
import numpy as np
import time

# Import Qiskit classes
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
IBMQ.load_accounts()
IBMQ.backends()
device = IBMQ.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map
gate_times = [
    ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
    ('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
    ('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
    ('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
    ('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
    ('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
    ('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)
]
noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)

import sys, os
#sys.path.append('/Users/atanteck/Desktop/project/qiskit-ignis/qiskit/ignis')
from verification import tomography as tomo
from mitigation import measurement as mc
import matplotlib.pyplot as plt 

N = 5
def rand_angles():
	return tuple(2 * np.pi * np.random.random(3) - np.pi)

#Add measurement noise
noise_model = noise.NoiseModel()
#for qi in range(2):
#    read_err = noise.errors.readout_error.ReadoutError([[0.75, 0.25],[0.1,0.9]])
#    noise_model.add_readout_error(read_err,[qi])

# Create a state preparation circuit
F_uncorr = []
F_corr = []
for i in range(N):
	q2 = QuantumRegister(2)
	bell = QuantumCircuit(q2)
	for j in range(2):
		bell.u3(*rand_angles(),q2[j])
	bell.h(q2[0])
	bell.cx(q2[0], q2[1])

	job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
	psi_bell = job.result().get_statevector(bell)
	qst_bell = tomo.state_tomography_circuits(bell, q2)


	    
	#generate the calibration circuits
	meas_calibs, state_labels = mc.measurement_calibration(qr=q2)

	backend = Aer.get_backend('qasm_simulator')
	qobj_cal = qiskit.compile(meas_calibs, backend=backend, shots=15000)
	qobj_tomo = qiskit.compile(qst_bell, backend=backend, shots=15000)

	job_cal = backend.run(qobj_cal, noise_model=noise_model)
	job_tomo = backend.run(qobj_tomo, noise_model=noise_model)

	meas_fitter = mc.MeasurementFitter(job_cal.result(),state_labels)
	tomo_counts_bell = tomo.tomography_data(job_tomo.result(), qst_bell,efficient=True)

	# Generate fitter data and reconstruct density matrix
	probs_bell, basis_matrix_bell, weights_bell = tomo.fitter_data(tomo_counts_bell)

	#no correction
	rho_bell = tomo.state_cvx_fit(probs_bell, basis_matrix_bell, weights_bell)
	F_bell = state_fidelity(psi_bell, rho_bell)
	F_uncorr.append(F_bell)
	print('Fit Fidelity (no correction) =', F_bell)

	#correct data
	probs_bell = meas_fitter.apply(probs_bell, method='least_squares')
	rho_bell = tomo.state_cvx_fit(probs_bell, basis_matrix_bell, weights_bell)
	F_bell = state_fidelity(psi_bell, rho_bell)
	F_corr.append(F_bell)
	print('Fit Fidelity (w/ correction) =', F_bell)
plt.plot(F_uncorr,'ro',label='Uncorrected')
plt.plot(F_corr,'ko',label='Corrected')
plt.legend()
plt.grid()
plt.xlabel("Trial")
plt.xlim([0,N])
plt.ylabel("fidelity")
plt.show()