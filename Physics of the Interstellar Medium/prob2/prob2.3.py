"""
Function: Solution to Problem 3.1 in Statistics & Numerical Methods.
Usage:    python3.11 prob3.1.py
Version:  Last edited by Cha0s_MnK on 2024-11-11 (UTC+08:00).
"""

###########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENT(S) #
###########################################

from config import *

nu31    = c / 4960.295e-10
nu32    = c / 5008.240e-10
nu43    = c / 4364.436e-10
A31     = 6.951e-3
A32     = 2.029e-2
A43     = 1.685
A41     = 2.255e-01
Omega30 = 0.243
Omega31 = 0.243 * 3
Omega32 = 0.243 * 5
Omega40 = 0.0321
g3      = 5

# Electron densities
n_es = [1e9, 1e10, 1e11]  # in m^-3

# Temperature range
T = np.linspace(5000, 20000, 500)  # in Kelvin

######################
# HELPER FUNCTION(S) #
######################

def calc_ratio(T, n_e):
    term1 = (Omega30 / Omega40) * (A41 / A43 + 1) * np.exp(h * nu43 / (k_B * T)) + 1
    term2 = A31 * nu31 + A32 * nu32
    term3 = A31 + A32 + n_e * np.sqrt(2 * np.pi * (h / 2 / np.pi)**4 / (k_B * m_e**3)) * (Omega30 + Omega31 + Omega32) / g3 / np.sqrt(T)
    return term1 * term2 / term3 / nu43

#################
# MAIN FUNCTION #
#################

def main():
    # plot and save
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi = 2 * DPI_MIN)
    for n_e in n_es:
        ax.plot(T, calc_ratio(T=T, n_e=n_e), label=r'$n_e$' + f' = {n_e:.0e}' + r' m$^{-3}$')
    set_fig(ax = ax,
            title = 'Observed Emissivity Ratio over Temperature for Different Electron Number Densities',
            xlabel = r'Temperature $T$ (K)', ylabel = 'Observed emissivity ratio', ylog=True)
    save_fig(fig = fig, name = f'prob2.3')

if __name__ == "__main__":
    main()