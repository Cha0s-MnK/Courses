"""
Function: Python script template.
Usage:    python3.9 prob1.4.py
Version:  Last edited by Cha0s_MnK on 2024-10-15 (UTC+08:00).
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

a_vec    = [0.01, 0.4, 1, 3, 10]
us       = np.linspace(-10, 10, 1000)
a_lim    = 0.0001
figsizeX = 10
figsizeY = 6

######################
# HELPER FUNCTION(S) #
######################

# Voigt function H(a, u)
def Voigt_func(a, u):
    integrand   = lambda y: np.exp(- y ** 2) / ((u - y) ** 2 + a ** 2)
    integral, _ = quad(integrand, -np.inf, np.inf)
    return a / np.pi * integral

# approximate form for small a (Gaussian core and Lorentzian wing approximation)
def approx_Voigt_func(a, u, u_0):
    if np.abs(u) < u_0:
        # Gaussian-like core for small u
        return np.exp(-u**2) + a / (np.sqrt(np.pi) * u**2)
    else:
        # Lorentz-like wing for large u
        return np.exp(-u**2) + a / (np.sqrt(np.pi) * u**2)

# Define the equation as a function of u_0
def eq_u_0(u_0, a):
    return u_0**2 * np.exp(-u_0**2) - a / np.sqrt(np.pi)

#################
# MAIN FUNCTION #
#################

def main():
    # problem 1.4.1

    # calculate and plot H(a, u) for different values of a
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    for a in a_vec:
        plt.plot(us, [Voigt_func(a, u) for u in us], label=f'$a$ = {a}')

    # plot settings and save the plot as a PNG file
    plt.title(r'Voigt-Hjerting function $H(a, u)$ for different values of $a$')
    plt.xlabel(r'$u$')
    plt.ylabel(r'$H(a, u)$')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob1.4.1.png")

    # problem 1.4.2

    # Calculate the Voigt-Hjerting function and its approximation for limit a and plot the results
    u_0      = fsolve(eq_u_0, 1.0, args=(a_lim)) # initial guess = 1.0
    print(f"u_0 = {u_0[0]:.6f}")
    u_0      = 1.8
    H        = [Voigt_func(a=a_lim, u=u) for u in us]
    H_approx = [approx_Voigt_func(a=a_lim, u=u, u_0=u_0) for u in us]
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.plot(us, H, label='numerical Voigt-Hjerting function', color='blue')
    plt.plot(us, H_approx, linestyle='--', label='approximate Voigt-Hjerting function', color='red')

    # plot settings and save the plot as a PNG file
    plt.title(r'Voigt-Hjerting function comparison for $a = 0.01$')
    plt.xlabel(r'$u$')
    plt.ylabel(r'$H(a, u)$')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob1.4.2.png")

    # problem 1.4.3

    # calculate and plot the relative error
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.plot(us, np.abs((np.array(H) - np.array(H_approx)) / np.array(H)), label=r'relative error $\varepsilon$', color='blue')
    plt.axvline(x=u_0, color='red', linestyle="--", label='criterion')
    plt.axvline(x=-u_0, color='red', linestyle="--")

    # plot settings and save the plot as a PNG file
    plt.title(r'Relative error of the approximate Voigt-Hjerting function for $a = 0.01$')
    plt.xlabel(r'$u$')
    plt.ylabel(r'relative error $\varepsilon$')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("prob1.4.3.png")

if __name__ == "__main__":
    main()