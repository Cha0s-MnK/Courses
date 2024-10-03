"""
Function: Plot plasma and Larmor frequencies for a range of electron densities or B-field strengths.
Usage:    python3.9 prob1.1.py
Version:  Last edited by Cha0s_MnK on 2024-10-03 (UTC+08:00).
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

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():

if __name__ == "__main__":
    main()
