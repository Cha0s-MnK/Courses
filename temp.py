"""
Function: Python script template.
Usage:    python3.9 temp.py
Version:  Last edited by Cha0s_MnK on 2024-10-13 (UTC+08:00).
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

pc = 3.0856775814913673e16 # [m]

###################
# SET ARGUMENT(S) #
###################

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    n_e = 5e4
    l   = 4e5 * pc
    sigma_T = 8 * np.pi * e**4 / (3 * m_e**2 * c**4)
    N   = n_e * sigma_T * l
    y = 0.00278
    T_e = y * m_e * c **2 / (4 * N * k_B)
    print(T_e)

if __name__ == "__main__":
    main()
