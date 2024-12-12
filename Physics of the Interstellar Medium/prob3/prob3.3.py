"""
Function: Solution to Problem 3.3 in Physics of the Interstellar Medium.
Usage:    python3.11 prob3.3.py
Version:  Last edited by Cha0s_MnK on 2024-12-11 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

k1    = FLOAT(8.4e-10) # [cm³·s⁻¹]
k2    = FLOAT(1.01e-9) # [cm³·s⁻¹]
k3    = FLOAT(4.3e-7)  # [cm³·s⁻¹]
k4    = FLOAT(6.4e-10) # [cm³·s⁻¹]
k5    = FLOAT(7.48e-8) # T₂ = 100 K; [cm³·s⁻¹]
k6    = FLOAT(3.5e-10) # [s⁻¹]
n_e   = FLOAT(1e4)     # [m⁻³]
n_H   = FLOAT(1e8)     # [m⁻³]
n_H2  = FLOAT(5e7)     # [m⁻³]
n_H3p = FLOAT(5)       # [m⁻³]
n_O   = FLOAT(4e4)     # [m⁻³]

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    # problem 3.3.1
    n_OHp = k1 * n_O * n_H3p / (k2 * n_H2) # [m⁻³]
    print(f"n_OHp = {n_OHp:e} (m^-3)")

    # problem 3.3.2
    n_H2Op = k1 * n_O * n_H3p / (k3 * n_e + k4 * n_H2) # [m⁻³]
    print(f"n_H2O = {n_H2Op:e} (m^-3)")

    # problem 3.3.3
    n_OH = (0.2 * k3 / k6 * n_e + 0.74 * k4 / k6 * n_H2) * n_H2Op # [m⁻³]
    print(f"n_OH = {n_OH:e} (m^-3)")
    print(f"n_OH/n_H = {n_OH / n_H:e}")

    # problem 3.3.4
    print(f"ratio = {k3 * n_e / (3 * k4 * n_H2)}")

if __name__ == "__main__":
    main()