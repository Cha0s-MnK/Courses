"""
Function: Solution to Problem 5.1 in Statistics & Numerical Methods.
Usage:    python3.11 prob5.1.py
Version:  Last edited by Cha0s_MnK on 2024-11-27 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

n  = INT(8)
k  = INT(5)
p0 = FLOAT(0.25)

######################
# HELPER FUNCTION(S) #
######################

def calc1():
    p_hat = k / n
    z     = (p_hat - p0) * math.sqrt(n / (p0 * (1 - p0)))
    # use the CDF of the standard normal distribution
    return 1 - stats.norm.cdf(z)

def calc2():
    def integrand(p):
        return p**5 * (1 - p)**3
    result, _ = integrate.quad(integrand, 0, 0.25)
    return math.factorial(9) / (math.factorial(5) * math.factorial(3)) * result

#################
# MAIN FUNCTION #
#################

def main():
    print(f"sigma_p_hat = {math.sqrt(p0 * (1 - p0))}")
    print(f"z           = {(k / n - p0) * math.sqrt(n / (p0 * (1 - p0)))}")
    print(f"Result of the integral in 5.1.1: {calc1()}")
    print(f"Result of the integral in 5.1.2: {calc2()}")

if __name__ == "__main__":
    main()