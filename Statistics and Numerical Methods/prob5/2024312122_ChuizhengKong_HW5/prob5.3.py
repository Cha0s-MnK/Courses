"""
Function: Solution to Problem 5.3 in Statistics & Numerical Methods.
Usage:    python3.11 prob5.3.py
Version:  Last edited by Cha0s_MnK on 2024-12-08 (UTC+08:00).
"""

############################################
# CONFIGURE ENVIRONMENT & SET PARAMETER(S) #
############################################

from config import *

N     = INT(1e6) # number of particles
r_s   = 20.0     # scale radius [kpc]
C     = 10.0     # concentration parameter
r_max = 30.0     # maximum radius [kpc]
H0    = 70.0     # Hubble constant [km·s⁻¹·Mpc⁻¹]

######################
# HELPER FUNCTION(S) #
######################

def calc_delta_c(C):
    return 200.0 * C**3 / (3 * (np.log(1 + C) + 1 / (1 + C) - 1))

def calc_G_mod():
    return G * M_sun / (kpc * 1e6)  # pc * (km/s)^2 / M_sun

def calc_Fr_coef(x):
    return np.log(1 + x) + 1 / (1 + x) - 1

#################
# MAIN FUNCTION #
#################

def main():
    # calculate parameters
    delta_c = calc_delta_c(C)
    H0_mod  = H0 / 1e3 # Hubble constant [km·s⁻¹·kpc⁻¹]
    G_mod   = calc_G_mod()
    rho_c   = (3 * H0_mod**2) / (8 * np.pi * G_mod)
    Fr_coef = 1 / calc_Fr_coef(x = r_max / r_s)
    M_max   = 4 * np.pi * rho_c * delta_c * r_s**3 / Fr_coef
    print(f"delta_c = {delta_c}")
    print(f"H0      = {H0_mod} km·s⁻¹·kpc⁻¹")
    print(f"G       = {G_mod} pc·(km/s)²·M☉⁻¹")
    print(f"rho_c   = {rho_c} M☉·kpc⁻³")
    print(f"Fr_coef = {Fr_coef}")
    print(f"M_max   = {M_max} M☉")

    # define the radius array for CDF calculation
    rs = np.linspace(0.0, r_max, INT(3e6))[1:] # avoid r = 0 to prevent lg(0)

    # calculate CDF F(r)
    Frs = Fr_coef * calc_Fr_coef(x = rs / r_s)

    # create inverse CDF interpolation function
    Frs_inv = interp1d(Frs, rs, kind='linear', bounds_error=False, fill_value=(rs[0], rs[-1]))

    # sample radii using inverse CDF
    rs_sample = Frs_inv(np.random.uniform(0.0, 1.0, N))

    # calculate radial density distribution
    bins         = np.linspace(0.1, r_max, 1999) # avoid r = 0 to prevent lg(0)
    bin_centres  = 0.5 * (bins[:-1] + bins[1:])
    dFrs, _      = np.histogram(rs_sample, bins=bins) # bin the sampled radii
    dVs          = (4 / 3) * np.pi * (bins[1:]**3 - bins[:-1]**3) # volume element of spherical shells
    rhos_sampled = M_max * dFrs / dVs / N

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi = 2 * DPI_MIN)
    ax.plot(np.log10(bin_centres), np.log10(rhos_sampled), label='Monte Carlo sampling', drawstyle='steps-mid')
    rhos_analy = rho_c * delta_c * r_s**3 / (rs * (rs + r_s)**2) # analytic NFW profile
    ax.plot(np.log10(rs), np.log10(rhos_analy), label='analytic')

    set_fig(ax = ax, title = r'Logarithmic Radial Density Distribution of the NFW Profile',
            xlabel = r'Logarithmic radius lg$\displaystyle \frac{r}{\mathrm{kpc}}$',
            ylabel = r'Logarithmic radial density lg$\displaystyle \frac{\rho_\Lambda(r)}{M_\odot \ \mathrm{kpc}^{-3}}$',
            xlim=[-1, np.log10(r_max)], ylim=[5.5, 9.5])
    save_fig(fig = fig, name = f'fig5.3')

if __name__ == "__main__":
    main()