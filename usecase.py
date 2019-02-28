import numpy as np

import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import qiskit.ignis.verification.tomography as tomo

from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise

from qiskit.ignis.verification.tomography.fitters.pure_state_mle_fit import pure_state_mle_fit, pure_state_mle_fit_density_matrix
from qiskit.ignis.test_tomography import *

from scipy.linalg import sqrtm


# Suppose we have some quantum circuit, which generates a state |psi>. We are interested in the coefficients of psi.
# Unfortunatly, due to noise effects, on a real quantum computer we can only generate rho = (1-epsilon)|psi><psi| + epsilon rho_error
# with epsilon << 1. How do we find a good estimate of |psi>
#
# Using usual quantum state tomography, we can could find rho and would estimate |psi> to be the eigenvector of rho which belongs to
# the larges eigenvalue. Using pure state tomography, we can find an estimate of |psi> directly from the measurements. This requires 
# fewer operators to be measured then for a full state tomography for the same accuraccy.


# Consider the Bell states |psi1> = 1/sqrt(2)(|00>+|11>) and |psi2> = 1/sqrt(2)(|00>-|11>) and rho = (1-epsilon)|psi1><psi1| + epsilon|psi2><psi2|

#Psi1
# Setup circuit to generate |psi1>
qr1 = QuantumRegister(2)
bell1 = QuantumCircuit(qr1)
bell1.ry(np.pi/2,qr1[0])
bell1.rx(-np.pi/2,qr1[1])
#bell1.h(qr1[0])
#bell1.cx(qr1[0], qr1[1])
#bell1.rx(-np.pi/2, qr1[0])
#bell1.ry(np.pi/2, qr1[1])

# get state |psi1> using statevector_simulator
job = qiskit.execute(bell1, Aer.get_backend('statevector_simulator'))
psi1 = job.result().get_statevector(bell1)
print("|psi1> = {}".format(psi1))

def get_expectation_values(tomo_counts, shots=5000):
    #operators = ['II', 'IX', 'IY', 'IZ', 'XI', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    operators = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    #operators = ['XY']
    probs = []
    basis_matrix=[]
    for s in operators:
        #print("Tomo counts: ",tomo_counts)
        prob, mat = compute_expectation(2, s, tomo_counts, shots)
        #print("operator: {}".format(s))
        #print("exp: {}".format(prob))
        #print("mat: {}".format(mat))
        probs.append(prob)
        basis_matrix.append(mat)
    return probs, basis_matrix

# get expectation values of operators ... and convert the to the right format
qst_bell1 = tomo.state_tomography_circuits(bell1, qr1)
job = qiskit.execute(qst_bell1, Aer.get_backend('qasm_simulator'), shots=5000)
tomo_counts_psi1 = tomo.tomography_data(job.result(), qst_bell1)
#probs_psi1, basis_matrix, _ = tomo.fitter_data(tomo_counts_psi1)
probs_psi1, basis_matrix = get_expectation_values(tomo_counts_psi1)


#Psi 2
# Setup circuit to generate |psi2>
qr2 = QuantumRegister(2)
bell2 = QuantumCircuit(qr2)
bell2.y(qr2[0])
bell2.h(qr2[0])
bell2.cx(qr2[0], qr2[1])

# get state |psi2> using statevector_simulator
job = qiskit.execute(bell2, Aer.get_backend('statevector_simulator'))
psi2 = job.result().get_statevector(bell2)
print("|psi2> = {}".format(psi2))


# get expectation values of operators ... and convert the to the right format
qst_bell2 = tomo.state_tomography_circuits(bell2, qr2)
job = qiskit.execute(qst_bell2, Aer.get_backend('qasm_simulator'), shots=5000)
tomo_counts_psi2 = tomo.tomography_data(job.result(), qst_bell2)
#probs_psi2, _ , _ = tomo.fitter_data(tomo_counts_psi2) # the basis_matrix is the same for both states
probs_psi2, basis_matrix = get_expectation_values(tomo_counts_psi2)

# Generate probabilities of rho
epsilon = 0
probs_rho = [(1-epsilon)*probs_psi1[i]+epsilon*probs_psi2[i] for i in range(len(probs_psi1))]

# function for calculating inner product between states
def inner_product(psi, phi):
    return np.abs(sum([psi[i].conj()*phi[i] for i in range(len(psi))]))**2

# Full state tomography (maximum likelyhood approach)
print("=== Full state tomography ===")
rho_full_mle = tomo.state_mle_fit(probs_rho, basis_matrix)
eigenvalues, eigenvectors = np.linalg.eig(rho_full_mle)
print("Eigenvalues of rho: {}".format(eigenvalues))
print("Guess for psi: {}".format(eigenvectors[0]))
print("Fidelity: {}".format(inner_product(psi1, eigenvectors[0])))


# Pure state tomography (maximum likelyhood approach)
print("=== Pure state tomography ===")
psi_guess,_ = pure_state_mle_fit(probs_rho, basis_matrix)
print("Guess for psi: {}".format(psi_guess))
print("Fidelity: {}".format(inner_product(psi1, psi_guess)))
ddd


# On a noisy simulator:
print("=== On Noisy simulator ===")
qr = QuantumRegister(2)
circuit = QuantumCircuit(qr)
params = np.random.rand(12)
circuit.u3(params[0], params[1], params[2], qr[0])
circuit.u3(params[3], params[4], params[5], qr[1])
circuit.cx(qr[0], qr[1])
circuit.u3(params[6], params[7], params[8], qr[0])
circuit.u3(params[9], params[10], params[11], qr[1])


job = qiskit.execute(circuit, Aer.get_backend('statevector_simulator'))
psi = job.result().get_statevector(circuit)
print("psi: {}".format(psi))


IBMQ.enable_account('a6140115a9d2692b8a711d0f31fdc92b7ae793865719445a78fc210220de52765da18d0eab774d245fbc9e5f40c94e99d9c536f9f1749acd7b902b3bd9931dd5')
device = IBMQ.get_backend('ibmqx4')
properties = device.properties()
coupling_map = device.configuration().coupling_map

gate_times = [
    ('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
    ('cx', [1, 0], 678)]

# Construct the noise model from backend properties
# and custom gate times
noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)


# Get the basis gates for the noise model
basis_gates = noise_model.basis_gates

# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

print(noise_model)

qst= tomo.state_tomography_circuits(circuit, qr)
job = qiskit.execute(qst, Aer.get_backend('qasm_simulator'),noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates, shots=5000)
tomo_counts = tomo.tomography_data(job.result(), qst)
#probs_psi, basis_matrix, _ = tomo.fitter_data(tomo_counts)
probs_psi, basis_matrix = get_expectation_values(tomo_counts)


# Full state tomography (maximum likelyhood approach)
print("=== Full state tomography ===")
rho_full_mle = tomo.state_mle_fit(probs_psi, basis_matrix)
eigenvalues, eigenvectors = np.linalg.eig(rho_full_mle)
print("Eigenvalues of rho: {}".format(eigenvalues))
print("Guess for psi: {}".format(eigenvectors[0]))
print("Fidelity: {}".format(inner_product(psi, eigenvectors[0])))


# Pure state tomography (maximum likelyhood approach)
print("=== Pure state tomography ===")
psi_guess,_ = pure_state_mle_fit(probs_psi, basis_matrix)
print("Guess for psi: {}".format(psi_guess))
print("Fidelity: {}".format(inner_product(psi, psi_guess)))

