"""
Function: Statistics & Numerical Methods, Problem 1.4
Usage:    python3.9 prob1.4.py
Version:  Last edited by Cha0s_MnK on 2024-10-05 (UTC+08:00).
"""

#########################
# CONFIGURE ENVIRONMENT #
#########################

import numpy as np

######################
# HELPER FUNCTION(S) #
######################

def calc_machine_precision(data_type: type):
    """
    Problem 1.1.1: Calculate the machine precision (epsilon_m/ε_m) for a given floating-point data type.
    
    "The machine precision ε_m, which can be considered as the smallest number ε_m such that 1+ε_m still
    evaluates to something different from 1." So we start with ε_m = 1 and repeatedly halve it until 1+ε_m is
    numerically equal to 1 in the given precision. The last value before 1+ϵ_m equals 1 is taken as the machine
    precision ϵ_m.

    Parameter(s):
    - data_type (type):      The floating-point data type (e.g. np.float32, np.float64) to evaluate.

    Return(s):
    - epsilon_m (data_type): The machine precision for the specified data type.
    """
    epsilon_m = data_type(1)
    one       = data_type(1)
    while one + epsilon_m != one:
        epsilon_m_last = epsilon_m
        epsilon_m      = data_type(epsilon_m) / data_type(2)
    return epsilon_m_last

def calc_smallest_positive(data_type: type):
    """
    Problem 1.1.2: Calculate the smallest positive number for a given floating-point data type.

    We start with f_min=1 and repeatedly halve it until the value becomes zero in the given precision. The last
    non-zero value before underflowing to zero is taken as f_min.

    Parameter(s):
    - data_type (type):  The floating-point data type (e.g. np.float32, np.float64) to evaluate.

    Return(s):
    - f_min (data_type): The smallest positive number for the specified data type.
    """
    f_min = data_type(1)
    while f_min > data_type(0):
        f_min_last = f_min
        f_min      = data_type(f_min) / data_type(2)
    return f_min_last

#################
# MAIN FUNCTION #
#################

def main():
    # calculate for single precision (32-bit) and double precision (64-bit)
    epsilon_m_single = calc_machine_precision(np.float32)
    epsilon_m_double = calc_machine_precision(np.float64)
    f_min_single     = calc_smallest_positive(np.float32)
    f_min_double     = calc_smallest_positive(np.float64)

    # print results
    print(f"Machine precision epsilon_m (single precision/np.float32):    {epsilon_m_single}")
    print(f"Machine precision epsilon_m (double precision/np.float64):    {epsilon_m_double}")
    print(f"Smallest positive number f_min (single precision/np.float32): {f_min_single}")
    print(f"Smallest positive number f_min (double precision/np.float64): {f_min_double}")

if __name__ == "__main__":
    main()