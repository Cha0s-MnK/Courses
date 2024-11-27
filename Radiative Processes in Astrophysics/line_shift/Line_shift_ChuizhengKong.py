"""
Function: Calculate the observed wavelength of the O VIII Ly-α line
Usage:    python3.11 Line_shift_ChuizhengKong.py
Version:  Last edited by Cha0s_MnK on 2024-11-24 (UTC+08:00).
"""

#######################################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS & SET OPTIONS #
#######################################################

import math

# physical constants (reference: https://en.wikipedia.org/wiki/List_of_physical_constants)
c         = 2.99792458e8        # speed of light [m·s⁻¹]

# arguments
lambda0 = 1.897e-9 # rest wavelength of O VIII Ly-α line [m]
z       = 0.3      # cosmological redshift
v       = 0.1 * c  # outflow velocity (10% of the speed of light) [m·s⁻¹]

#################
# MAIN FUNCTION #
#################

def main():
    beta      = v / c # relativistic Doppler shift factor due to the outflow velocity
    lambda_ob = lambda0 * (1 + z) * math.sqrt((1 + beta) / (1 - beta))

    print(f"The observed wavelength of the O VIII Ly-α line is {lambda_ob} metre.")

if __name__ == "__main__":
    main()