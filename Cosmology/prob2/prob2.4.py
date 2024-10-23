"""
Function: Solution to Problem 2.4 in Cosmology.
Usage:    python3.9 prob2.4.py
Version:  Last edited by Cha0s_MnK on 2024-10-22 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     'mathtext.default': 'regular',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'text.usetex': True})
from scipy.integrate import quad

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

Omega_r0      = 8e-5
Omega_m0      = 0.25 - 8e-5
Omega_Lambda0 = 0.75 + 8e-5
H_0           = 7e4 / 3.0856775814913673e22 # Hubble constant [s⁻¹]
zs            = [30, 20, 6, 2, 1, 0.1]
Gyr           = 3.15576e16 # [s]

######################
# HELPER FUNCTION(S) #
######################

def Hubble_parameter(z):
    return H_0 * np.sqrt(Omega_r0 * (1 + z)**4 + Omega_m0 * (1 + z)**3 + Omega_Lambda0)

def integrand_func(z):
    """ integrand function f(x) """
    return 1.0 / ((1 + z) * Hubble_parameter(z))
 
def calc_universe_age(z):
    t, err = quad(integrand_func, z, np.inf, args=())
    return t

#################
# MAIN FUNCTION #
#################

def main():
    # problem 2.4.1
    z     = Omega_m0 / Omega_r0 - 1
    t_Gyr = calc_universe_age(z=z) / Gyr
    print(f"when z = {z:>.3e}, t(z) = {t_Gyr:.3e} Gyr")

    # problem 2.4.2
    z = (Omega_Lambda0 / Omega_m0)**(1/3) - 1
    t_Gyr = calc_universe_age(z=z) / Gyr
    print(f"when z = {z:>.3e}, t(z) = {t_Gyr:.3e} Gyr")

    # problem 2.4.3
    for z in zs:
        t_Gyr = calc_universe_age(z=z) / Gyr
        print(f"when z = {z:>.3e}, t(z) = {t_Gyr:.3e} Gyr")


if __name__ == "__main__":
    main()
