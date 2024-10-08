"""
Function: Statistics & Numerical Methods, Problem 1.4
Usage:    python3.9 prob1.4.py
Version:  Last edited by Cha0s_MnK on 2024-10-08 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np

######################
# HELPER FUNCTION(S) #
######################

def Hilbert_matrix(N: int, data_type: type):
    """Construct an n x n Hilbert matrix with specified data type."""
    A = np.zeros((N, N), dtype=data_type)
    for i in range(N):
        for j in range(N):
            A[i, j] = 1.0 / (i + j + 1)
    return A

def construct_b(n, dtype=np.float64):
    """Construct vector b such that the exact solution x = [1, ..., 1]^T."""
    b = np.zeros(n, dtype=dtype)
    for i in range(n):
        b_i = 0.0
        for j in range(n):
            b_i += 1.0 / (i + j + 1)
        b[i] = b_i
    return b

def Cholesky_decomposition(A):
    """Perform Cholesky decomposition on a positive definite matrix A."""
    N = A.shape[0]
    L = np.zeros_like(A)
    for i in range(N):
        for j in range(i):
            sum_k = np.dot(L[i,:j], L[j,:j])
            if i == j:
                L[i,j] = np.sqrt(A[i,i] - sum_k)
            else:
                L[i,j] = (A[i,j] - sum_k) / L[j,j]
    return L

def solve_cholesky(L, b):
    """Solve Ax = b using the Cholesky factor L."""
    # Forward substitution to solve Ly = b
    y = np.zeros_like(b)
    n = L.shape[0]
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
    # Backward substitution to solve L^T x = y
    x = np.zeros_like(b)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(L.T[i,i+1:], x[i+1:])) / L.T[i,i]
    return x

def relative_error(x_numeric, x_exact):
    """Compute the relative error between numeric and exact solutions."""
    return np.linalg.norm(x_numeric - x_exact, np.inf) / np.linalg.norm(x_exact, np.inf)

#################
# MAIN FUNCTION #
#################

def main():
    n = 5
    x_exact = np.ones(n)
    for dtype in [np.float32, np.float64]:
        print(f"\nData type: {dtype}")
        A = hilbert_matrix(n, dtype=dtype)
        b = construct_b(n, dtype=dtype)
        L = cholesky_decomposition(A)
        x_numeric = solve_cholesky(L, b)
        error = relative_error(x_numeric, x_exact)
        print(f"Numeric solution x:\n{x_numeric}")
        print(f"Relative error: {error:e}")

if __name__ == "__main__":
    main()