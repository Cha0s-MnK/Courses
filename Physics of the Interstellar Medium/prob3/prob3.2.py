"""
Function: Solution to Problem 3.2 in Physics of the Interstellar Medium.
Usage:    python3.11 prob3.2.py
Version:  Last edited by Cha0s_MnK on 2024-12-10 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

a_max    = FLOAT(0.3)         # [μm]
beta_n_H = FLOAT(1e-6 * 0.01) # [μm·yr⁻¹]
dotM_d   = FLOAT(1.3e-4)      # [M☉·yr⁻¹]
p        = FLOAT(3.5)

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    # problem 3.2.4
    tau_s = (4 - p) * a_max / ((p - 1) * (5 - p) * beta_n_H)
    print(f"tau_s = {tau_s} [yr]")
    print(f'M_d   = {tau_s * dotM_d} [M_odot]')

if __name__ == "__main__":
    main()