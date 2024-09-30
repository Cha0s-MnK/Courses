"""
Function: Plot the plasma and Larmor's frequencies for a range of electron densities or B-field strengths.
Usage:    python3.9 Plasma_freq_ChuizhengKong.py
Version:  Last edited by Cha0s_MnK on 2024-09-30 (UTC+08:00).
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

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8       # speed of light [m·s⁻¹]
e         = 1.602176634e-19    # elementary charge [C]
epsilon_0 = 8.854187817e-12    # vacuum permittivity [F·m⁻¹]
h         = 6.62607015e-34     # Planck constant [J·s]
k_B       = 1.380649e-23       # Boltzmann constant [J·K⁻¹]
m_e       = 9.109383713928e-31 # electron mass [kg]

###################
# SET ARGUMENT(S) #
###################

# Frequency range of LHAASO telescope (in Hz)
LHAASO_min_freq = 1e8  # 100 MHz
LHAASO_max_freq = 1e9  # 1 GHz

# Define range of electron densities (in m^-3) and magnetic field strengths (in Tesla)
ns_e = np.logspace(6, 12, 100)  # electron number density range: from 1e6 to 1e12 m^-3
Bs   = np.logspace(-9, -3, 100)  # magnetic field strength range: from 1 nT to 1 mT

######################
# HELPER FUNCTION(S) #
######################

def calc_plasma_freq(n_e: np.ndarray) -> np.ndarray:
    """
    Calculate the plasma frequency for a given electron density using the following formula:
    omega_p = sqrt((4·pi·n_e·e²)/m_e)

    Parameter(s):
    - n_e     (np.ndarray): electron number density [m⁻³]

    Return(s):
    - omega_p (np.ndarray): Plasma (angular) frequency [s⁻¹]
    """
    return np.sqrt((4 * np.pi * n_e * e**2) / m_e)

def calc_Larmor_freq(B: np.ndarray) -> np.ndarray:
    """
    Calculate the Larmor frequency for a given magnetic field strength using the following formula:
    omega_L = e·B/m_e·c

    Parameter(s):
    - B       (np.ndarray): magnetic field strength [T]

    Return(s):
    - omega_L (np.ndarray): Larmor (angular) frequency [s⁻¹]
    """
    return (e * B) / (m_e * c)

#################
# MAIN FUNCTION #
#################

def main():
    # plot Planck spectrum for various temperatures
    plt.figure(figsize=(figsizeX, figsizeY))
    # Plot plasma frequency vs. electron density
    plt.loglog(electron_densities, plasma_frequencies, label='Plasma Frequency', color='blue')

    # Plot Larmor frequency vs. magnetic field strength
    plt.loglog(B_fields, larmor_frequencies, label='Larmor Frequency', color='green')

    # Plot the LHAASO telescope frequency range
    plt.fill_betweenx([1e6, 1e12], LHAASO_min_freq, LHAASO_max_freq, color='gray', alpha=0.3, label='LHAASO Frequency Range')

    # plot settings
    plt.xlabel(r'Wavelength $\lambda$ (m)')
    plt.ylabel(r'Specific Intensity $I_\lambda$ (W·m$^{-3}$·sr$^{-1}$)')
    plt.xlim(10 ** lgXmin, 10 ** lgXmax)
    plt.ylim(Ymin, Ymax)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.xlabel('Electron Density (m$^{-3}$) / Magnetic Field Strength (T)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Plasma and Larmor Frequencies with LHAASO Telescope Range')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # save the plot as a PDF file
    plt.savefig("Plasma_freq_ChuizhengKong.pdf")

if __name__ == "__main__":
    main()
