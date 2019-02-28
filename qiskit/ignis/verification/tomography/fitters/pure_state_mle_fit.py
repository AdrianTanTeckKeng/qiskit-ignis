
import numpy as np
from scipy.optimize import minimize

def pure_state_mle_fit(data, basis_matrix, weights = None):
    # We are solving the least squares fit: minimize ||a * x - b ||_2
    # where:
    #   a is the matrix of measurement operators
    #   b is the vector of expectation value data for each projector
    #   x is the vectorized density matrix (or Choi-matrix) to be fitted
    a = basis_matrix
    b = np.array(data)

    # Optionally apply a weights vector to the data and projectors
    if weights is not None:
        w = np.array(weights)
        a = w[:, None] * a
        b = w * b

    n = int(np.sqrt(len(a[0])))

    def cost_function(psi):
        s = 0
        for i in range(len(b)):
            #projector_matrix = np.array([[ a[i][j+n*k] for j in range(n)] for k in range(n)])
            #expectation_value = np.dot(psi, projector_matrix.dot(psi))
            expectation_value = 0
            for j in range(n):
                for k in range(n):
                    expectation_value+=(psi[2*j]*psi[2*k]+psi[2*j+1]*psi[2*k+1])*a[i][j+n*k]
            s += (b[i] - expectation_value)**2
        return s

    opt_res = minimize(cost_function, [1]+[0]*(2*n-1))
    norm = np.sqrt(sum([x**2 for x in opt_res.x]))
    psi = []
    for i in range(n):
        psi.append((opt_res.x[2*i]+1j*opt_res.x[2*i+1])/norm)
    return psi, opt_res.fun
