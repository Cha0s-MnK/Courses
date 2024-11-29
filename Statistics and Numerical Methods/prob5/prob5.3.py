"""
Function: Solution to Problem 5.3 in Statistics & Numerical Methods.
Usage:    python3.11 prob5.3.py
Version:  Last edited by Cha0s_MnK on 2024-11-28 (UTC+08:00).
"""

############################################
# CONFIGURE ENVIRONMENT & SET PARAMETER(S) #
############################################

from config import *

N = int(1e6)           # Number of particles
r_s = 20.0             # Scale radius in kpc
C = 10.0               # Concentration parameter
r_max = 30.0           # Maximum radius in kpc
H0 = 70.0              # Hubble constant in km/s/Mpc

######################
# HELPER FUNCTION(S) #
######################

def delta_c(C):
    numerator = (200.0 / 3.0) * C**3
    denominator = np.log(1 + C) - C / (1 + C)
    return numerator / denominator

# Cumulative mass function M(s)
def M_s(s):
    return np.log(1 + s) - s / (1 + s)

#################
# MAIN FUNCTION #
#################

def main():
    # Calculate delta_c
    delta_c_value = delta_c(C)

    # Convert H0 to km/s/kpc
    H0_kpc = H0 / 1000.0  # km/s/kpc

    # Gravitational constant in units of kpc * (km/s)^2 / M_sun
    G = 4.30091e-6  # kpc (km/s)^2 / M_sun

    # Calculate critical density in M_sun/kpc^3
    rho_c = (3 * H0_kpc**2) / (8 * np.pi * G)

    # Characteristic density rho_s
    rho_s = rho_c * delta_c_value

    # Dimensionless radius s = r / r_s
    s_max = r_max / r_s

    # Generate an array of s values
    s = np.linspace(1e-5, s_max, 100000)

    # Total mass within s_max
    M_tot = M_s(s_max)

    # Cumulative distribution function F(s)
    F_s = M_s(s) / M_tot

    # Create interpolation function for inverse CDF
    inverse_cdf = interp1d(F_s, s, kind='linear')

    # Generate random numbers uniformly distributed between 0 and 1
    u_random = np.random.uniform(0, 1, N)

    # Use inverse transform sampling to get s values
    s_random = inverse_cdf(u_random)

    # Calculate r values
    r_random = s_random * r_s

    # Compute theoretical density profile
    r_theory = np.logspace(np.log10(0.01), np.log10(r_max), 1000)
    s_theory = r_theory / r_s
    rho_theory = rho_s / (s_theory * (1 + s_theory)**2)

    # Compute particle density profile
    bins = np.logspace(np.log10(0.01), np.log10(r_max), 50)
    counts, edges = np.histogram(r_random, bins=bins)
    shell_volumes = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    particle_density = counts / shell_volumes

    # Calculate bin centers
    bin_centers = (edges[1:] + edges[:-1]) / 2

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(r_theory), np.log10(rho_theory), label='NFW Profile (Equation 5.3.1)', linewidth=2)
    plt.scatter(np.log10(bin_centers), np.log10(particle_density), color='red', s=10, label='Particle Realization')
    plt.xlabel(r'$\log_{10}[r/\mathrm{kpc}]$')
    plt.ylabel(r'$\log_{10}\left[ \rho_\Lambda(r) / (M_\odot\, \mathrm{kpc}^{-3}) \right]$')
    plt.title('Logarithmic Radial Density Distribution')

if __name__ == "__main__":
    main()