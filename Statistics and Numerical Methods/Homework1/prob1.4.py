"""
Function: Statistics & Numerical Methods, Problem 1.4
Usage:    python3.9 prob1.4.py
Version:  Last edited by Cha0s_MnK on 2024-10-09 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np

###################
# SET ARGUMENT(S) #
###################

np.set_printoptions(precision=3)

data_types      = [np.float32, np.float64]
err_c           = 0.5                      # threshold for relative error (50%)
Nstart          = 5
Nmax            = 32
Ns_unstable     = {}
PRINT_WIDTH     = 80

######################
# HELPER FUNCTION(S) #
######################

def Hilbert_matrix(N: int, data_type: type = float) -> np.ndarray:
    """
    Construct an N x N Hilbert matrix of a specified data type. Hilbert matrix is a well-known example of
    ill-conditioned matrices, which looks like this:

    A_ij = 1 / (i + j + 1)

    Parameter(s):
    - N (int):          The dimension of the matrix (number of rows or columns).
    - data_type (type): The data type of the matrix elements (default is float).

    Return(s):
    - A (np.ndarray):   An N x N Hilbert matrix of the specified data type.
    """
    A = np.zeros((N, N), dtype=data_type)
    for i in range(N):
        for j in range(N):
            A[i, j] = 1.0 / (i + j + 1)
    return A

def Hilbert_vector(N: int, data_type: type = float) -> np.ndarray:
    """
    Construct a Hilbert vector b such that the exact solution x = [1, ..., 1]^T when solving Ax = b, where A is
    a Hilbert matrix. The vector b is calculated as:

    b_i = sum_{j=0}^{N-1} 1 / (i + j + 1)

    Parameter(s):
    - N (int):          The dimension of the vector.
    - data_type (type): The data type of the vector elements (default is float).

    Returns:
    - b (np.ndarray):   An Hilbert vector b of length N of the specified data type.
    """
    b = np.zeros(N, dtype=data_type)
    for i in range(N):
        for j in range(N):
            b[i] += 1.0 / (i + j + 1)
    return b

def Cholesky_decompose(A):
    """
    Perform Cholesky decomposition on a real symmetric positive definite matrix A. In other words, compute the
    lower triangular matrix L such that:
    
    A = L * L^T

    Parameter(s):
    - A (np.ndarray): A real symmetric positive definite matrix of size N x N.

    Return(s):
    - L (np.ndarray): A lower triangular matrix of size N x N.
    """
    N = A.shape[0]
    L = np.zeros_like(A)
    for i in range(N):
        for j in range(i+1):
            sum_k = np.dot(L[i,:j], L[j,:j])
            if j == i:
                L[i,j] = np.sqrt(A[i,i] - sum_k)
            else:
                L[i,j] = (A[i,j] - sum_k) / L[j,j]
    return L

def solve_Cholesky(A, b):
    """
    Solve the linear system Ax = b using the Cholesky decomposition. First perform Cholesky decomposition to
    obtain L. Then solve Ly = b for y (forward substitution) and finally solve L^T x = y for x (backward
    substitution).

    Parameter(s):
    - A (np.ndarray): A real symmetric positive definite matrix of size N x N.
    - b (np.ndarray): A real vector of length N.

    Return(s):
    - x (np.ndarray): The solution vector of length N.
    """
    L = Cholesky_decompose(A)
    N = L.shape[0]

    # solve Ly = b for y by forward substitution
    y = np.zeros_like(b)
    for i in range(N):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    # solve L^T x = y for x by backward substitution
    x = np.zeros_like(b)
    for i in reversed(range(N)):
        x[i] = (y[i] - np.dot(L[i+1:,i], x[i+1:])) / L[i,i]

    return x

def calc_relative_error(x_exact, x_numeric, norm=np.inf):
    """
    Compute the relative error between numeric and exact solutions using the specified norm.
    
    Parameters:
    - x_exact (vector-like):   The exact solution vector.
    - x_numeric (vector-like): The numeric (approximate) solution vector.
    - norm (int or float):     The order of the norm (default is infinity).

    Returns:
    - err_relative (float):   The relative error between x_numeric and x_exact using the specified norm.
    """
    # check if inputs are 1D (vector-like)
    if x_exact.ndim != 1 or x_numeric.ndim != 1:
        raise ValueError("x_exact and x_numeric must be 1D vectors.")
    
    # check if shapes are compatible
    if x_exact.shape != x_numeric.shape:
        raise ValueError("x_exact and x_numeric must have the same shape.")

    # compute the numerator and denominator using the specified norm
    numerator   = np.linalg.norm(x_numeric - x_exact, ord=norm)
    denominator = np.linalg.norm(x_exact, ord=norm)
    
    if denominator == 0:
        if numerator == 0:
            return 0.0
        else:
            raise ZeroDivisionError("The exact solution has zero norm. So the relative error is undefined.")
    
    return numerator / denominator

def solve_Hilbert_problem(N, norm=np.inf, data_type: type = float):
    """
    Analyzes the relative error in solving a system of linear equations
    with a Hilbert matrix using Cholesky decomposition for a specified data type.
    
    Parameters:
    - N (int): Size of the Hilbert matrix.
    - data_type (type): The data type to use for the computations (np.float32 or np.float64).
    
    This function returns the relative error of the computed solution.
    """
        
    # solve Ax=b using Cholesky decomposition
    x_exact   = np.ones(N)
    x_numeric = solve_Cholesky(Hilbert_matrix(N, data_type), Hilbert_vector(N, data_type))
        
    # calculate the relative error using specified norm
    err_r = calc_relative_error(x_exact=x_exact, x_numeric=x_numeric, norm=norm)

    # print the results
    print("-" * PRINT_WIDTH)
    print(f"Data type:          {data_type.__name__}")
    print(f"Hilbert matrix size N         = {N}")
    print(f"Exact solution      x_exact   = {np.ones(N)}") # vector of ones
    print(f"Numeric solution    x_numeric = {x_numeric}")
    print(f"Relative error      err_r     = {err_r * 100:.3f}%")
    print("-" * PRINT_WIDTH)
    return err_r

def calc_condition_number(A, norm=np.inf) -> np.float64:
    """
    Calculate the condition number of matrix A based on the specified norm.

    Parameter(s):
    - A (np.ndarray): The input square matrix.

    Returns:
    - cond_number (float): The condition number of A based on the infinity norm.
    """
    # compute the inverse of A
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix A is singular and cannot be inverted.")

    # compute the specified norm of A and A_inv
    normA    = np.linalg.norm(A, ord=norm)
    normAinv = np.linalg.norm(Ainv, ord=norm)

    # calculate and return the condition number
    return normA * normAinv

#################
# MAIN FUNCTION #
#################

def main():
    # problem 1.4.2
    print("\n" + "=" * PRINT_WIDTH)
    print(f"{'PROBLEM 1.4.2':^{PRINT_WIDTH}}")
    print("=" * PRINT_WIDTH)
    for data_type in data_types:
        solve_Hilbert_problem(N=Nstart, norm=2, data_type=data_type)

    # problem 1.4.3
    print("\n" + "=" * PRINT_WIDTH)
    print(f"{'PROBLEM 1.4.3':^{PRINT_WIDTH}}")
    print("=" * PRINT_WIDTH)
    for data_type in data_types:
        print("\n" + "-" * PRINT_WIDTH)
        print(f"{'Analyzing data type: ' + data_type.__name__:^{PRINT_WIDTH}}")
        print("-" * PRINT_WIDTH)
        flag_unstable = False
        for N in range(Nstart, Nmax + 1):
            try:
                err_r = solve_Hilbert_problem(N=N, norm=2, data_type=data_type)
                if err_r > err_c:
                    Ns_unstable[data_type] = N
                    print(f"Method becomes unstable at N = {N} for data type {data_type.__name__}.")
                    flag_unstable = True
                    break  # Stop testing larger N values for this data type
            except Exception as ERROR:
                print(f"An unexpected error occurred at N = {N}: {ERROR}")
                Ns_unstable[data_type] = N
                flag_unstable = True
                break  # Stop testing larger N values for this data type
        if not flag_unstable:
            print(f"Method remains stable up to N = {Nmax} for data type {data_type.__name__}.\n")

    # problem 1.4.4
    print("\n" + "=" * PRINT_WIDTH)
    print(f"{'PROBLEM 1.4.4':^{PRINT_WIDTH}}")
    print("=" * PRINT_WIDTH)
    print(f"{'N':>4} | {'Data Type':>10} | {'Condition Number (infinity-norm)':>20}")
    print("-" * 54)
    for data_type in data_types:
        for N in [3, 6, 9, 12]:
            print(f"{N:>4} | {data_type.__name__:>10} | {calc_condition_number(Hilbert_matrix(N, data_type=data_type), norm=np.inf):>20.4e}")

if __name__ == "__main__":
    main()