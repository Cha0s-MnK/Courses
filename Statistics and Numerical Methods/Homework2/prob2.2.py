"""
Function: Solution to Problem 2.2 in Statistics & Numerical Methods.
Usage:    python3.9 prob2.2.py
Version:  Last edited by Cha0s_MnK on 2024-10-22 (UTC+08:00).
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
from scipy.signal import find_peaks

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

# Load the data from the file and Extract the columns: position (x), brightness (y), and noise (dy)
data = np.loadtxt('stars.dat')
x    = data[:, 0]
y    = data[:, 1]
dy   = data[:, 2]

######################
# HELPER FUNCTION(S) #
######################



#################
# MAIN FUNCTION #
#################

def main():
    # problem 2.2.1

    # Plot the brightness with error bars
    plt.errorbar(x, y, yerr=dy, fmt='o', label='Observed Data', capsize=3)
    plt.xlabel('Position (x)')
    plt.ylabel('Brightness (y)')
    plt.title('1D Image of Stars with 1σ Error Bars')
    plt.grid(True)
    plt.legend()

    # Simple peak finding to estimate number of stars (based on local maxima)
    # Find the peaks (stars)
    peaks, _ = find_peaks(y, height=0)

    # Plot the data with peaks highlighted
    plt.errorbar(x, y, yerr=dy, fmt='o', label='Observed Data', capsize=3)
    plt.plot(x[peaks], y[peaks], 'rx', label=f'Estimated Stars (N={len(peaks)})')
    plt.xlabel('Position (x)')
    plt.ylabel('Brightness (y)')
    plt.title(f'Number of Stars: {len(peaks)}')
    plt.grid(True)
    plt.legend()

    # Print the number of stars identified
    print(f"Estimated number of stars (N): {len(peaks)}")

    # problem 2.4.2
    # Define the Gaussian point-spread function (PSF)
def psf(x, xs):
    return np.exp(-(x - xs)**2 / 2) / np.sqrt(2 * np.pi)

# Define the model for the brightness profile
def model(params, x, n_stars):
    A = params[:n_stars]  # Amplitudes
    xs = params[n_stars:]  # Star positions
    y_model = np.zeros_like(x)
    for j in range(n_stars):
        y_model += A[j] * psf(x, xs[j])
    return y_model

# Define the residuals (to minimize)
def residuals(params, x, y, dy, n_stars):
    y_model = model(params, x, n_stars)
    return (y - y_model) / dy

# Initial guesses for star positions and amplitudes (based on peak positions)
n_stars = len(peaks)
A_init = y[peaks]  # Initial guess for amplitudes
xs_init = x[peaks]  # Initial guess for star positions
params_init = np.concatenate([A_init, xs_init])

# Perform the least-squares optimization
result = least_squares(residuals, params_init, args=(x, y, dy, n_stars))

# Extract the optimized parameters
A_opt = result.x[:n_stars]
xs_opt = result.x[n_stars:]

# Plot the observed data with the fitted model
plt.errorbar(x, y, yerr=dy, fmt='o', label='Observed Data', capsize=3)
plt.plot(x, model(result.x, x, n_stars), 'r-', label='Fitted Model')
plt.xlabel('Position (x)')
plt.ylabel('Brightness (y)')
plt.title('Fitted Model vs Observed Data')
plt.grid(True)
plt.legend()
plt.show()

# Print the optimized parameters
print(f"Optimized Amplitudes (A): {A_opt}")
print(f"Optimized Star Positions (x_s): {xs_opt}")

if __name__ == "__main__":
    main()
