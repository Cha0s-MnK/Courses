"""
Function: Python script template.
Usage:    python3.9 prob1.2.py
Version:  Last edited by Cha0s_MnK on 2024-10-18 (UTC+08:00).
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

b          = 2e4        # [m·s⁻¹]
gamma_ul   = 4.7e8      # [s⁻¹]
lambda_0   = 1.21567e-7 # [m]
W_lambda_g = 2e-10      # [m]

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    # problem 1.5.1
    tau_0 = c * W_lambda_g / (np.sqrt(np.pi) * b * lambda_0)
    print(f'Linear part assumption:      tau_0    = {tau_0}')
    tau_0 = np.exp(c**2 * W_lambda_g**2 / (4 * b**2 * lambda_0**2))
    print(f'Flat part assumption:        tau_0    = {tau_0}')
    tau_damp = 0.455 * b * (1 + 0.2 * np.log(b / 1e3))
    print(f'Lyman alpha:                 tau_damp = {tau_damp}')
    tau_0 = np.sqrt(np.pi) * c**2 * W_lambda_g**2 / (b * lambda_0**3 * gamma_ul)
    print(f'Square-root part assumption: tau_0    = {tau_0}')
    n_l   = 1.9e22 * (W_lambda_g * 1e10)**2
    print(f'Lyman alpha:                 n_l      = {n_l}')

if __name__ == "__main__":
    main()