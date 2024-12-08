"""
Function: Solution to Problem 5.2 in Cosmology.
Usage:    python3.11 prob5.2.py
Version:  Last edited by Cha0s_MnK on 2024-12-08 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

c_km       = c / 1e3 # speed of light [km·s⁻¹]
H0         = 70      # Hubble constant [km·s⁻¹·Mpc⁻¹]
Omega_b    = 0.03
Omega_r    = 8e-5
Omega_m    = 0.3
z_decouple = 1100

######################
# HELPER FUNCTION(S) #
######################

def H(z):
    """Hubble parameter H as a function of redshift z"""
    return H0 * np.sqrt(Omega_r * (1+z)**4 + Omega_m * (1+z)**3 + 1 - Omega_m - Omega_r)

def c_s(z):
    """sound speed c_s as a function of redshift z"""
    R = 3 * Omega_b / (4 * Omega_r * (1+z))
    return c / np.sqrt(3 * (1 + R))

#################
# MAIN FUNCTION #
#################

def main():
    D_SH_c, _ = quad(lambda z: c_s(z) / H(z), z_decouple, np.inf) # [Mpc]
    D_A_c, _  = quad(lambda z: c / H(z), 0, z_decouple)           # [Mpc]
    theta_SH = D_SH_c / D_A_c
    l_SH     = np.pi / theta_SH

    print(f"D_SH_c   = {D_SH_c}")
    print(f"D_A_c    = {D_A_c}")
    print(f"theta_SH = {theta_SH}")
    print(f"l_SH     = {l_SH}")

if __name__ == "__main__":
    main()