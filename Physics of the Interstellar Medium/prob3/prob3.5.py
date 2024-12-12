"""
Function: Solution to Problem 3.5 in Physics of the Interstellar Medium.
Usage:    python3.11 prob3.5.py
Version:  Last edited by Cha0s_MnK on 2024-12-12 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

Delta_theta = FLOAT(57.5 / 180 * pi) #
DM          = FLOAT(200 * pc * 1e6)  # [m⁻²]
nu1         = FLOAT(1610e6)          # [s⁻¹]
nu2         = FLOAT(1660e6)          # [s⁻¹]

######################
# HELPER FUNCTION(S) #
######################

#################
# MAIN FUNCTION #
#################

def main():
    # problem 3.5.1
    RMmin = Delta_theta * nu1**2 * nu2**2 / (c**2 * (nu2**2 - nu1**2))
    print(f"RMmin = {RMmin:e} m^-2")
    RMnext = np.abs(Delta_theta - pi) * nu1**2 * nu2**2 / (c**2 * (nu2**2 - nu1**2))
    print(f"RMnext = {RMnext:e} m^-2")

    # problem 3.5.2
    coef  = 2 * pi * m_e**2 * c**4 / e**3
    print(f"coef = {coef:e} m^-2 Hz^-2")
    B_bar = 2 * pi * m_e**2 * c**4 * RMmin / (e**3 * DM)
    print(f"B_bar = {B_bar:e} T")

if __name__ == "__main__":
    main()