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

from verification import tomography as tomo
from verification.tomography.fitters.pure_state_mle_fit import *
from mitigation import measurement as mc


sgm_matrices = np.zeros([2,2,4],dtype=np.complex128)
sgm_matrices[0,0,0] = 1
sgm_matrices[1,1,0] = 1
sgm_matrices[0,1,1] = 1
sgm_matrices[1,0,1] = 1
sgm_matrices[0,1,2] = -1j
sgm_matrices[1,0,2] = 1j
sgm_matrices[0,0,3] = 1
sgm_matrices[1,1,3] = -1

def _findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter != ch]

def _indices(s):
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
	prob_array = np.zeros([2**nbits])
	for i in range(2**nbits):
		binary = '{0:b}'.format(i).zfill(nbits)
		try:
			if new_data[binary]:
				prob_array[i] = (new_data[binary]*1.0)/shots
		except:
			prob_array[i] = 0.
		for j in range(nbits):
			binary_array[i][j] = int(binary[j])
	key = key[-1::-1]
	binary_array = binary_array[:,_findOccurrences(key,'I')]
	binary_array = np.sum(binary_array,axis=1)
	expectation = 0
	for i in range(2**nbits):
		if binary_array[i]&1 == 0:
			expectation += prob_array[i]
		else:
			expectation -= prob_array[i]
	mat = sgm_matrices[:,:,_indices(key[0])]
	for i in range(1,nbits):
		mat = np.kron(mat,sgm_matrices[:,:,_indices(key[i])])
	mat = np.reshape(mat,[4**nbits])
	return expectation,mat
