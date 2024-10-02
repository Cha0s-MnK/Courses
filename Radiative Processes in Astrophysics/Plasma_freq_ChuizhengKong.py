"""
Function: Plot plasma and Larmor frequencies for a range of electron densities or B-field strengths.
Usage:    python3.9 Plasma_freq_ChuizhengKong.py
Version:  Last edited by Cha0s_MnK on 2024-10-02 (UTC+08:00).
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

# figure size
figsizeX = 9
figsizeY = 6

# X limits
lgX1min = 0
lgX1max = 16
lgX2min = -16
lgX2max = 0

# frequency range of FAST [Hz]
freqFASTmin = 7.0e7
freqFASTmax = 3.0e9

# electron number density and magnetic field strength ranges
ns_e = np.logspace(lgX1min, lgX1max, 100)  # electron (number) density range: 1.0e0, 1.0e1, ..., 1.0e16 [m⁻³]
Bs   = np.logspace(lgX2min, lgX2max, 100)  # magnetic field strength range: 1.0e-16, 1.0e-15, ..., 1.0e0 [T]

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
    # plot settings 1
    fig, ax1 = plt.subplots(figsize=(figsizeX, figsizeY))
    ax1.set_xlabel(r'electron number density $n_e$ (m$^{-3}$)')
    ax1.set_ylabel(r'angular frequencies $\omega$ (s$^{-1}$)')
    ax1.set_title('Plasma and Larmor frequencies with FAST range')
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax2 = ax1.twiny()
    ax2.set_xlabel(r'magnetic field strength $B$ (T)')

    # plot the plasma, Larmor frequencies and the FAST frequency range
    ax1.loglog(ns_e, calc_plasma_freq(ns_e), color='red', label=r'plasma frequencies $\omega_\mathrm{p}$')
    ax2.loglog(Bs, calc_Larmor_freq(Bs), color='green', label=r'Larmor frequencies $\omega_\mathrm{L}$')
    ax2.axhspan(freqFASTmin, freqFASTmax, color='gray', alpha=0.3, label='FAST frequency range')

    # plot settings 2
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    plt.tight_layout()

    # save the plot as a PDF file
    plt.savefig("Plasma_freq_ChuizhengKong.pdf")

if __name__ == "__main__":
    main()
