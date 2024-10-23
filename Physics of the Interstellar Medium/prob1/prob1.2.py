"""
Function: Python script template.
Usage:    python3.9 prob1.2.py
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

figsizeX = 10
figsizeY = 6
Gamma    = 2e-33                       # photoelectric heating rate by dust [J·s⁻¹]
p        = 2e-14                       # pressure [N·m⁻²]
Ts       = np.logspace(1, 4.301, 1000) # temperature range from 10 to 20000 K
x        = 1e-3                        # fractional ionization
Ymin     = 1e-45
Ymax     = 1e-37

######################
# HELPER FUNCTION(S) #
######################

def Lambda_CII__e(x, T):
    return 3.1e-37 * x * (T / 100)**(-0.5) * np.exp(-91.2 / T)

def Lambda_CII__H(x, T):
    return 5.2e-40 * (T / 100)**0.13 * np.exp(-91.2 / T)

def Lambda_OI__H(x, T):
    return 4.1e-40 * (T / 100)**0.42 * np.exp(-228 / T)

def Lambda_Ly_alpha__e(x, T):
    return 6e-32 * x * (T / 1e4)**(-0.5) * np.exp(-1.18e5 / T)

def Lambda_local(x, T):
    return Lambda_CII__e(x, T) + Lambda_CII__H(x, T) + Lambda_OI__H(x, T) + Lambda_Ly_alpha__e(x, T)

def Lambda_low_metallicity(x, T):
    return 0.1 * Lambda_CII__e(x, T) + 0.1 * Lambda_CII__H(x, T) + 0.1 * Lambda_OI__H(x, T) + Lambda_Ly_alpha__e(x, T)

#################
# MAIN FUNCTION #
#################

def main():
    # problem 1.2.1

    # plot
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.loglog(Ts, Lambda_CII__e(x=x, T=Ts), label=r'[C II] 158$\mu$m with electrons', linestyle='--')
    plt.loglog(Ts, Lambda_CII__H(x=x, T=Ts), label=r'[C II] 158$\mu$m with H atoms', linestyle='--')
    plt.loglog(Ts, Lambda_OI__H(x=x, T=Ts), label=r'[O I] 63$\mu$m with H atoms', linestyle='--')
    plt.loglog(Ts, Lambda_Ly_alpha__e(x=x, T=Ts), label=r'Ly$\alpha$ with electrons', linestyle='--')
    plt.loglog(Ts, Lambda_local(x=x, T=Ts), label=r'net cooling function', color='blue')

    # plot settings and save the plot as a PNG file
    plt.ylim([Ymin, Ymax])
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'Cooling Rate (J $\cdot$ cm$^3$ $\cdot$ s$^{-1}$)')
    plt.title('Cooling functions $\Lambda(x, T)$ for $x = 10^{-3}$ in the local ISM')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob1.2.1.png")

    # problem 1.2.2

    # plot
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.loglog(Ts, 0.1 * Lambda_CII__e(x=x, T=Ts), label=r'[C II] 158$\mu$m with electrons', linestyle='--')
    plt.loglog(Ts, 0.1 * Lambda_CII__H(x=x, T=Ts), label=r'[C II] 158$\mu$m with H atoms', linestyle='--')
    plt.loglog(Ts, 0.1 * Lambda_OI__H(x=x, T=Ts), label=r'[O I] 63$\mu$m with H atoms', linestyle='--')
    plt.loglog(Ts, Lambda_Ly_alpha__e(x=x, T=Ts), label=r'Ly$\alpha$ with electrons', linestyle='--')
    plt.loglog(Ts, Lambda_low_metallicity(x=x, T=Ts), label=r'net cooling function', color='blue')

    # plot settings and save the plot as a PNG file
    plt.ylim([Ymin, Ymax])
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'Cooling Rate (J$\cdot$cm$^3 \cdot$s$^{-1}$)')
    plt.title('Cooling functions $\Lambda(x, T)$ for $x = 10^{-3}$ in the low-metallicity ISM')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob1.2.2.png")

    # problem 1.2.3

    # plot
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    ns_T = Gamma / Lambda_low_metallicity(x=x, T=Ts)
    plt.loglog(Ts, ns_T, label=r'equilibrium density $n(T)$', color='blue')

    # plot settings and save the plot as a PNG file
    plt.xlabel(r'temperature $T$ (K)')
    plt.ylabel(r'equilibrium density $n(T)$ (m$^{-3}$)')
    plt.title('Equilibrium density for heating and cooling balance in the low-metallicity ISM')
    plt.legend()
    plt.grid(True, which="both", linestyle=':')
    plt.tight_layout()
    plt.savefig("prob1.2.3.png")

    # problem 1.2.4

    # plot
    ns_T = Gamma / Lambda_low_metallicity(x=x, T=Ts)
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.loglog(Ts, ns_T * k_B * Ts, label='equilibrium curve', color='blue')
    plt.axhline(y=p, color='red', linestyle=':', label=r'pressure $p = 2 \times 10^{-14}$ N$\cdot$m$^{-2}$')

    # plot settings and save the plot as a PNG file
    plt.xlabel(r'temperature $T$ (K)')
    plt.ylabel(r'pressure $p$ (N$\cdot$m$^{-2}$)')
    plt.title('Two-phase ISM equilibrium diagram in the low-metallicity ISM')
    plt.legend()
    plt.grid(True, which="both", linestyle='--')
    plt.tight_layout()
    plt.savefig("prob1.2.4-1.png")

    # plot
    ns_T = Gamma / Lambda_local(x=x, T=Ts)
    plt.figure(figsize=(figsizeX, figsizeY), dpi=512)
    plt.loglog(Ts, ns_T * k_B * Ts, label='equilibrium curve', color='blue')
    plt.axhline(y=p, color='red', linestyle=':', label=r'pressure $p = 2 \times 10^{-14}$ N$\cdot$m$^{-2}$')

    # plot settings and save the plot as a PNG file
    plt.xlabel(r'temperature $T$ (K)')
    plt.ylabel(r'pressure $p$ (N$\cdot$m$^{-2}$)')
    plt.title('Two-phase ISM equilibrium diagram in the local ISM')
    plt.legend()
    plt.grid(True, which="both", linestyle='--')
    plt.tight_layout()
    plt.savefig("prob1.2.4-2.png")

if __name__ == "__main__":
    main()
