import numpy as np

import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer import noise
import qiskit.ignis.verification.tomography as tomo

from qiskit.ignis.verification.tomography.fitters.pure_state_mle_fit import pure_state_mle_fit, pure_state_mle_fit_density_matrix

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
bell1.h(qr1[0])
bell1.cx(qr1[0], qr1[1])

# get state |psi1> using statevector_simulator
job = qiskit.execute(bell1, Aer.get_backend('statevector_simulator'))
psi1 = job.result().get_statevector(bell1)
print("|psi1> = {}".format(psi1))

# get expectation values of operators ... and convert the to the right format
qst_bell1 = tomo.state_tomography_circuits(bell1, qr1)
job = qiskit.execute(qst_bell1, Aer.get_backend('qasm_simulator'), shots=5000)
tomo_counts_psi1 = tomo.tomography_data(job.result(), qst_bell1, efficient = True)
probs_psi1, basis_matrix, _ = tomo.fitter_data(tomo_counts_psi1)


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
tomo_counts_psi2 = tomo.tomography_data(job.result(), qst_bell2, efficient = True)
probs_psi2, _ , _ = tomo.fitter_data(tomo_counts_psi2) # the basis_matrix is the same for both states


# Generate probabilities of rho
epsilon = 0.3
probs_rho = [(1-epsilon)*probs_psi1[i]+epsilon*probs_psi2[i] for i in range(len(probs_psi1))]

# function for calculating inner product between states
def inner_product(psi, phi):
    return np.sqrt(sum([psi[i].conj()*phi[i] for i in range(len(psi))]))

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




