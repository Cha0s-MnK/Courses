"""
Configuration file for Python scripts used in collisional N-body dynamics simulation.

Usage:
    python3.9 Planck_spec_ChuizhengKong.py

Version:
    Last edited by Cha0s_MnK on 2024-09-28 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "Times New Roman",
    'mathtext.default': 'regular',
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})
plt.rcParams['text.usetex'] = True
from scipy.constants import h, c, k
k_B = k

"""
# physical constants
h   = 6.62607015e-34 # Planck constant [J·s]
c   = 2.99792458e8   # speed of light [m/s]
k_B = 1.380649e-23   # Boltzmann constant [J/K]
"""
###################
# SET ARGUMENT(S) #
###################

# figure size
figsizeX = 9
figsizeY = 6

# X and Y limits
lgXmin = -12
lgXmax = 1
Ymin   = 1e-13
Ymax   = 1e39

# JWST observable range
JWSTlambdaStart = 6.0e-7
JWSTlambdaStop  = 2.83e-5
JWSTposX        = 4.3e-6

######################
# HELPER FUNCTION(S) #
######################

def Planck_law(Lambda: np.ndarray, T: float) -> np.ndarray:
    """
    Calculate the specific intensity of a black body at a given wavelength and temperature.

    Parameter(s):
    - Lambda   (np.ndarray): wavelength array [m]
    - T        (float):      temperature [K]

    Return(s):
    - I_Lambda (np.ndarray): specific intensity [W/m^3]
    """
    return (2.0 * h * c**2) / (Lambda**5) / (np.exp((h * c) / (Lambda * k_B * T)) - 1.0)

#################
# MAIN FUNCTION #
#################

def main():
    # temperature and wavelength ranges
    Ts      = 3 * np.logspace(0, 8, num=9, base=10) # temperature range: 3.0e0, 3.0e1, ..., 3.0e8 K
    Lambdas = np.logspace(lgXmin, lgXmax, num=1000) # wavelength range: 1.0e-12, 1.0e-11, ..., 1.0e0 m

    # plot Planck spectrum for various temperatures
    plt.figure(figsize=(figsizeX, figsizeY))
    colours = plt.cm.tab10(np.linspace(0, 1, len(Ts)))
    for T, colour in zip(Ts, colours):
        plt.plot(Lambdas, Planck_law(Lambdas, T), label=f"{T:.1e} K", color=colour)

    # add JWST observable wavelength (6.0e-7 - 2.83e-5 m)
    plt.axvspan(JWSTlambdaStart, JWSTlambdaStop, color='gray', alpha=0.3, label='JWST observable range')
    plt.text(JWSTposX, plt.ylim()[1], f'({JWSTlambdaStart}, {JWSTlambdaStop})', color='black', fontsize=10, ha='center', va='bottom')

    # plot settings
    plt.xlabel(r'Wavelength $\lambda$ (m)')
    plt.ylabel(r'Specific Intensity $I_\lambda$ (W·m$^{-3}$)')
    plt.title('Planck Spectrum for Various Temperatures')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10 ** lgXmin, 10 ** lgXmax)
    plt.ylim(Ymin, Ymax)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    # save the plot as a PDF file
    plt.savefig("Planck_spec_ChuizhengKong.pdf")

if __name__ == "__main__":
    main()