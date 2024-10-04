"""
Function: Statistics & Numerical Methods, Problem 1.3
Usage:    python3.9 prob1.3.py
Version:  Last edited by Cha0s_MnK on 2024-10-04 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c = 2.99792458e8 # speed of light [m·s⁻¹]

###################
# SET ARGUMENT(S) #
###################

c          /= 1e3             # [m·s⁻¹] -> [km·s⁻¹]
redshifts   = [1.0, 3.0, 8.2]
varepsilon  = 1e-4            # desired relative precision

######################
# HELPER FUNCTION(S) #
######################

def integrand_func(x):
    """ integrand function f(x) """
    return 1.0 / np.sqrt(0.3 * (1 + x)**3 + 0.7)

def composite_Simpson(a: float, b: float, N: int) -> float:
    """
    Compute the numerical integral of a function using the composite Simpson formula

    Parameters:
    - a (float): lower limit of the numerical integral
    - b (float): upper limit of the numerical integral
    - N (int):   number of intervals (must be even)

    Returns:
    - I (float): approximation of the numerical integral
    """
    if N % 2 != 0:
        raise ValueError("Number of intervals (N) must be even.")

    x = np.linspace(a, b, N + 1)
    y = integrand_func(x)
    S = (y[0] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2]) + y[-1]) / 3
    return (b - a) / N * S

def adaptive_simpson(a: float, b: float, epsilon: float, N_max: int = int(1e7)) -> tuple[float, int]:
    """
    Adaptively compute the numerical integral of a function using the composite Simpson formula to achieve
    desired precision.

    Parameters:
    - a (float):          lower limit of the numerical integral
    - b (float):          upper limit of the numerical integral
    - varepsilon (float): desired relative precision
    - N_max (int):        maximum allowed number of intervals

    Returns:
    - I (float):          approximation of the numerical integral
    - N (int):            number of intervals used
    """
    N         = 2                          # start with 2 intervals
    I_last    = composite_Simpson(a, b, N) # compute the numerical integral I_N
    converged = False
    while not converged:
        N *= 2 # double the number of intervals
        if N > N_max:
            print("Maximum allowed number of intervals (N_max) reached.")
            break

        I = composite_Simpson(a, b, N) # recompute the numerical integral I_2N

        error_relative = abs(I - I_last) / 15.0 / abs(I) # estimate the relative error
        if error_relative < varepsilon:
            converged = True

        I_last = I
    return I, N

#################
# MAIN FUNCTION #
#################

def main():
    for z in redshifts:
        I, N = adaptive_simpson(0, z, varepsilon)
        d_c  = c / 67 * I  # co-moving distance [Mpc]
        print(f"Redshift            z      = {z}")
        print(f"Co-moving distance  d_c(z) = {d_c:.2f} Mpc")
        print(f"Number of intervals N      = {N}")
        print("-" * 40)

if __name__ == "__main__":
    main()
