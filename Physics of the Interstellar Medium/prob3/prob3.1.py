"""
Function: Solution to Problem 3.1 in Physics of the Interstellar Medium.
Usage:    python3.11 prob3.1.py
Version:  Last edited by Cha0s_MnK on 2024-12-09 (UTC+08:00).
"""

#########################################
# CONFIGURE ENVIRONMENT & SET ARGUMENTS #
#########################################

from config import *

a_0      = FLOAT(0.1e-6)  # [m]
a_max0   = FLOAT(0.28e-6) # [m]
a_max1   = FLOAT(0.25e-6) # [m]
a_max2   = FLOAT(0.35e-6) # [m]
as_max   = [FLOAT(0.25e-6), FLOAT(0.35e-6), FLOAT(0.45e-6), FLOAT(0.55e-6), FLOAT(0.65e-6)] # [m]
beta1    = FLOAT(1.5)
beta2    = FLOAT(2.0)
lambda_B = FLOAT(445e-9)  # [m]
lambda_V = FLOAT(551e-9)  # [m]
p_min    = FLOAT(3.0)
p_max    = FLOAT(4.0)
p1       = FLOAT(3.5)
ps       = np.linspace(p_min, p_max, 999)

######################
# HELPER FUNCTION(S) #
######################

def calc_sigma_e_eff1(a_max: FLOAT, beta: FLOAT, lambda_: FLOAT, p: FLOAT) -> FLOAT:
    C_1 = 2 / (beta - p + 3)
    C_2 = (pi * a_0 / (2 * lambda_))**beta
    C_3 = (a_max / a_0)**(beta - p + 3)
    return C_1 * C_2 * C_3

def calc_sigma_e_eff2(a_max: FLOAT, beta: FLOAT, lambda_: FLOAT, p: FLOAT) -> FLOAT:
    C1 = 2 / (3 - p)
    C2 = (a_max / a_0)**(3 - p)
    C3 = beta / (beta - p + 3)
    C4 = (2 * lambda_ / (pi * a_0))**(3 - p)
    return C1 * C2 - C1 * C3 * C4

def calc_R_V(a_max: FLOAT, beta: FLOAT, p: FLOAT) -> FLOAT:
    if a_max < a_max0:
        sigma_e_eff_V = calc_sigma_e_eff1(a_max=a_max, beta=beta, lambda_=lambda_V, p=p)
        sigma_e_eff_B = calc_sigma_e_eff1(a_max=a_max, beta=beta, lambda_=lambda_B, p=p)
    else:
        sigma_e_eff_V = calc_sigma_e_eff2(a_max=a_max, beta=beta, lambda_=lambda_V, p=p)
        sigma_e_eff_B = calc_sigma_e_eff2(a_max=a_max, beta=beta, lambda_=lambda_B, p=p)
    return sigma_e_eff_V / (sigma_e_eff_B - sigma_e_eff_V)

#################
# MAIN FUNCTION #
#################

def main():
    # problem 3.1.1
    print(f"pi * a / lambda < {np.pi * a_max0 / lambda_B}")
    print(f"sigma_e_eff (lambda_V) = {calc_sigma_e_eff1(a_max=a_max1, beta=beta1, lambda_=lambda_V, p=p1)}")

    # problem 3.1.2
    print(f"sigma_e (lambda_B) / sigma_e (lambda_V) = {(lambda_V / lambda_B)**beta1}")

    # problem 3.1.3
    print(f"R_V1 = {1 / ((lambda_V / lambda_B)**beta1 - 1)}")

    # problem 3.1.5
    sigma_e_eff_V = calc_sigma_e_eff2(a_max=a_max2, beta=beta2, lambda_=lambda_V, p=p1)
    sigma_e_eff_B = calc_sigma_e_eff2(a_max=a_max2, beta=beta2, lambda_=lambda_B, p=p1)
    print(f"sigma_e_eff (lambda_V) = {sigma_e_eff_V}")
    print(f"sigma_e_eff (lambda_B) = {sigma_e_eff_B}")
    print(f"R_V2 = {sigma_e_eff_V / (sigma_e_eff_B - sigma_e_eff_V)}")

    # problem 3.1.6
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi = 2 * DPI_MIN)
    for idx, a_max in enumerate(as_max):
        Rs_V = [calc_R_V(a_max=a_max, beta=beta2, p=p) for p in ps]
        ax.plot(ps, Rs_V, label=f'$a_{{\\mathrm{{max}}}} = {a_max}$ m')
    set_fig(ax = ax, title = f'$R_V$ vs $p$ for different $a_{{\\mathrm{{max}}}}$', xlabel = r'$p$',
            ylabel = r'$R_V$', xlim=[p_min, p_max])
    save_fig(fig = fig, name = f'fig3.1.6')

if __name__ == "__main__":
    main()