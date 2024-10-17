"""
Function: Statistics & Numerical Methods, Problem 2.1
Usage:    python3.9 prob2.1.py
Version:  Last edited by Cha0s_MnK on 2024-10-17 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     'mathtext.default': 'regular',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'text.usetex': True})
from scipy.integrate import quad
from scipy.optimize import fsolve

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8        # speed of light [m·s⁻¹]
e         = 1.602176634e-19     # elementary charge [C]
epsilon_0 = 8.854187817e-12     # vacuum permittivity [F·m⁻¹]
h         = 6.62607015e-34      # Planck constant [J·s]
k_B       = 1.380649e-23        # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31  # electron mass [kg]
m_p       = 1.6726219259552e-27 # proton mass [kg]

###################
# SET ARGUMENT(S) #
###################

figsizeX     = 10
figsizeY     = 6
N            = 20    # number of grid points
NumMaxIterat = 2000
NormOrder    = 2
l            = 1     # size of each grid cell
omega        = 1.5
varepsilon   = 1e-6

L            = N * l # Domain size [-L/2, L/2]
varphi_field = np.zeros((N+2, N+2))  # Initialize potential including ghost cells N+2 × N+2
X, Y         = np.meshgrid(np.linspace(-L/2 + l/2, L/2 - l/2, N), np.linspace(-L/2 + l/2, L/2 - l/2, N))

rho_field    = np.exp(-X**2 - Y**2) # Source term N × N

######################
# HELPER FUNCTION(S) #
######################

def enforceBCs(varphi_field):
    # enforce boundary conditions (BCs): varphi = 0 at boundaries

    varphi_field[0, :]  = - varphi_field[1, :] # left boundary
    varphi_field[-1, :] = -varphi_field[-2, :] # right boundary
    varphi_field[:, 0]  = -varphi_field[:, 1]  # bottom boundary
    varphi_field[:, -1] = -varphi_field[:, -2] # top boundary

def calcR(varphi_field):
    # calculate residual matrix R

    return rho_field - varphi_field[2:,1:-1] - varphi_field[:-2,1:-1] - varphi_field[1:-1,2:] \
           - varphi_field[1:-1,:-2] + 4 * varphi_field[1:-1,1:-1]

def calcAx(x):
    x_0padded = np.zeros((N+2, N+2)) # Create a zero-padded matrix of size (m+2, n+2)
    x_0padded[1:-1, 1:-1] = x # Copy the original matrix into the center of the padded matrix
    return x_0padded[2:, 1:-1] + x_0padded[:-2, 1:-1] + x_0padded[1:-1, 2:] + x_0padded[1:-1, :-2] - 4 * x

def solveJacobi(varphi_field):
    # Jacobi method

    varphi_field_new = varphi_field.copy()
    normRs           = []
    for k in range(NumMaxIterat):
        enforceBCs(varphi_field)
        R     = calcR(varphi_field=varphi_field)
        normR = np.linalg.norm(R, ord=NormOrder)
        normRs.append(normR)
        if normR < varepsilon:
            break
        varphi_field_new[1:-1,1:-1] = 0.25 * (varphi_field[2:,1:-1] + varphi_field[:-2,1:-1] +
                                      varphi_field[1:-1,2:] + varphi_field[1:-1,:-2] - rho_field)
        varphi_field                = varphi_field_new.copy()
    return varphi_field, normRs

def solveGS(varphi_field):
    # Gauss-Seidel (GS) method

    normRs = []
    for k in range(NumMaxIterat):
        enforceBCs(varphi_field)
        R     = calcR(varphi_field)
        normR = np.linalg.norm(R, ord=NormOrder)
        normRs.append(normR)
        if normR < varepsilon:
            break
        for i in range(1, N+1):
            for j in range(1, N+1):
                varphi_field[i,j] = 0.25 * (varphi_field[i+1,j] + varphi_field[i-1,j] + varphi_field[i,j+1]
                                    + varphi_field[i,j-1] - rho_field[i-1,j-1])
    return varphi_field, normRs

def solveSOR(varphi_field):
    # successive over-relaxation (SOR) method

    normRs = []
    for k in range(NumMaxIterat):
        enforceBCs(varphi_field)
        R     = calcR(varphi_field)
        normR = np.linalg.norm(R, ord=NormOrder)
        normRs.append(normR)
        if normR < varepsilon:
            break

        for i in range(1, N+1):
            for j in range(1, N+1):
                varphi_field[i,j] = (1 - omega) * varphi_field[i,j] + omega * 0.25 * (varphi_field[i+1,j]
                                    + varphi_field[i-1,j] + varphi_field[i,j+1] + varphi_field[i,j-1]
                                    - rho_field[i-1,j-1])

    return varphi_field, normRs

def solve_steepest_descent(varphi_field):
    # steepest descent method

    normRs = []
    for k in range(NumMaxIterat):
        enforceBCs(varphi_field)
        R     = calcR(varphi_field)
        normR = np.linalg.norm(R, ord=NormOrder)
        normRs.append(normR)
        if normR < varepsilon:
            break

        varphi_field[1:-1,1:-1] += np.sum(R**2) / np.sum(R * calcAx(x=R)) * R

    return varphi_field, normRs

def solve_conjugate_gradient(varphi_field):
    # conjugate gradient method

    normRs = []
    for k in range(NumMaxIterat):
        enforceBCs(varphi_field)
        R     = calcR(varphi_field)
        normR = np.linalg.norm(R, ord=NormOrder)
        normRs.append(normR)
        if normR < varepsilon:
            break

        if k == 0:
            P                    = R
            alpha_numerator      = np.sum(R**2)
        AP                       = calcAx(x=P)
        alpha                    = alpha_numerator / np.sum(P * AP)
        varphi_field[1:-1,1:-1] += alpha * P
        R                       -= alpha * AP
        alpha_numerator_new      = np.sum(R**2)
        P                        = R + alpha_numerator_new / alpha_numerator * P
        alpha_numerator          = alpha_numerator_new

    return varphi_field, normRs

#################
# MAIN FUNCTION #
#################

def main():
    # Run all methods and collect results
    methods = { 'Jacobi':             solveJacobi,
                'Gauss-Seidel':       solveGS,
                'SOR':                solveSOR,
                'steepest descent':   solve_steepest_descent,
                'conjugate gradient': solve_conjugate_gradient }
    results = {}
    for name, method in methods.items():
        print(f"Running {name} method...")
        varphi_field, normRs = method(np.zeros((N+2, N+2)))
        results[name] = {'varphi_field': varphi_field,
                         'normRs': normRs}
        print(f"{name} method converged in {len(normRs) - 1} iterations.")

    # plot
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    for name, data in results.items():
        plt.plot(data['normRs'], label=name)

    # plot settings and save the plot as a PNG file
    plt.yscale('log')
    plt.xlabel(r'number of iterations $n_\mathrm{iterat}$')
    plt.ylabel(r'Residual 2-norm $||R||_2$')
    plt.title('Residual 2-norm vs Number of Iterations')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob2.1.png")

if __name__ == "__main__":
    main()