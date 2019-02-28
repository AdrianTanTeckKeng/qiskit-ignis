# Needed for functions
import numpy as np
import time
import re

# Import Qiskit classes
import qiskit 
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
#sys.path.append('~/projects/ibm_hackathon/qiskit-ignis/qiskit/ignis')
from qiskit.ignis.verification import tomography as tomo

from qiskit.ignis.verification.tomography.fitters.pure_state_mle_fit import *
from qiskit.ignis.mitigation import measurement as mc

import matplotlib.pyplot as plt 

N = 1
sgm_matrices = np.zeros([2,2,4],dtype=np.complex128)
sgm_matrices[0,0,0] = 1
sgm_matrices[1,1,0] = 1
sgm_matrices[0,1,1] = 1
sgm_matrices[1,0,1] = 1
sgm_matrices[0,1,2] = -1j
sgm_matrices[1,0,2] = 1j
sgm_matrices[0,0,3] = 1
sgm_matrices[1,1,3] = -1

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter != ch]

def indices(s):
	if s == 'I':
		return 0
	elif s == 'X':
		return 1
	elif s == 'Y':
		return 2
	elif s == 'Z':
		return 3
	else:
		raise KeyError

def compute_expectation(nbits,key,data,shots):
	temp = key.replace('I','Z')
	new_key = ()
	for char in temp:
		new_key += (char,)
	new_data = data[new_key]
	binary = np.zeros([2**nbits,nbits],dtype=np.int32)
	binary_array = np.zeros([2**nbits,nbits],dtype=np.int32)
	prob_array = np.zeros([2**nbits],dtype=np.int32)
	for i in range(2**nbits):
		binary = '{0:b}'.format(i).zfill(nbits)
		try:
			if new_data[binary]:
				prob_array[i] = new_data[binary]/shots
		except:
			prob_array[i] = 0.
		for j in range(nbits):
			binary_array[i][j] = int(binary[j])
	binary_array = binary_array[:,findOccurrences(key,'I')]
	binary_array = np.sum(binary_array,axis=1)
	expectation = 0
	for i in range(2**nbits):
		if binary_array[i]&1 == 0:
			expectation += prob_array[i]
		else:
			expectation -= prob_array[i]
	mat = sgm_matrices[:,:,indices(key[0])]
	for i in range(1,nbits):
		mat = np.kron(mat,sgm_matrices[:,:,indices(key[i])])
	mat = np.reshape(mat,[4**nbits])
	return expectation,mat

def rand_angles():
	return tuple(2 * np.pi * np.random.random(3) - np.pi)

for i in range(0,N):
	nbits = 2
	q2 = QuantumRegister(nbits)
	bell = QuantumCircuit(q2)
	shots = 5000
	'''
	for j in range(2):
		bell.u3(*rand_angles(),q2[j])
	'''
	
	bell.h(q2[0])
	bell.cx(q2[0],q2[1])
	job = qiskit.execute(bell, Aer.get_backend('statevector_simulator'))
	psi_bell = job.result().get_statevector(bell)

	# Generate circuits and run on simulator
	t = time.time()
	qst_bell = tomo.state_tomography_circuits(bell, q2)
	
	job = qiskit.execute(qst_bell, Aer.get_backend('qasm_simulator'), shots=shots)
	print('Time taken:', time.time() - t)

	# Extract tomography data so that countns are indexed by measurement configuration
	# Note that the None labels are because this is state tomography instead of process tomography
	# Process tomography would have the preparation state labels there
	tomo_counts_bell = tomo.tomography_data(job.result(), qst_bell)
	expectation = compute_expectation(nbits,'IX',tomo_counts_bell,shots)
	# Generate fitter data and reconstruct density matrix

	probs_bell, basis_matrix_bell, weights_bell = tomo.fitter_data(tomo_counts_bell)
	rho_bell = pure_state_mle_fit_density_matrix(probs_bell, basis_matrix_bell, weights_bell)
	#rho_bell = rho_bell[0][0]
	print(rho_bell[0][0]/rho_bell[0][1])
	#print(type(rho_bell[0][0]))
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
