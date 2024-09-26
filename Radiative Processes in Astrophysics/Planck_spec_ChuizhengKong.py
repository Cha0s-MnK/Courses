"""
Configuration file for Python scripts used in collisional N-body dynamics simulation.

Usage:
    python3.9 Planck_spec_ChuizhengKong.py

Version:
    Last edited by Cha0s_MnK on 2024-09-26 (UTC+08:00).
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

# physical constants
h   = 6.62607015e-34 # Planck constant [J*s]
c   = 2.99792458e8   # speed of light [m/s]
k_B = 1.380649e-23   # Boltzmann constant [J/K]

###################
# SET ARGUMENT(S) #
###################

# X and Y limits
lgXmin = -12
lgXmax = 1
Ymin = 1e-13
Ymax = 1e39

######################
# HELPER FUNCTION(S) #
######################

def Planck_law(Lambda, T):
    """
    Calculate the spectral radiance of a black body at a given wavelength and temperature.
    
    Parameter(s):
    - Lambda (float): wavelength [m]
    - T      (float): temperature [K]
    
    Return(s):
    - Spectral radiance in erg/s/cm^2/cm/sr
    """
    return (2.0 * h * c**2) / (Lambda**5) / (np.exp((h * c) / (Lambda * k_B * T)) - 1.0)

#################
# MAIN FUNCTION #
#################

def main():
    # temperature and wavelength ranges
    Ts = [3 * 10**i for i in range(9)] # temperature range: 3e0, 3e1, 3e2, ..., 3e8 K
    Lambdas_m = np.logspace(lgXmin, lgXmax, num=1000) # wavelength range: 1e-12, 1e-11, 1e-10, ..., 1e0 m

    # plot
    plt.figure(figsize=(10, 7))
    colours = plt.cm.viridis(np.linspace(0, 1, len(Ts)))
    for T, colour in zip(Ts, colours):
        plt.plot(Lambdas_nm, Planck_law(Lambdas_m, T), label=f"{T} K", color=colour)

    # plot settings
    plt.xlabel("Wavelength / nm")
    plt.ylabel("Spectral Radiance (erg/s/cm²/cm/sr)")
    plt.title("Planck Spectrum for Various Temperatures")
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