import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
h   = 6.62607015e-27  # Planck constant in erg*s
c   = 2.99792458e10   # Speed of light in cm/s
k_B = 1.380649e-16    # Boltzmann's constant in erg/K

def Planck_law(Lambda, T):
    """
    Calculate the spectral radiance of a black body at a given wavelength and temperature.
    
    Parameters:
    - Lambda: wavelength [m]
    - T: temperature [K]
    
    Returns:
    - Spectral radiance in erg/s/cm^2/cm/sr
    """
    exponent = (h * c) / (Lambda * k_B * T)
    B_Lambda = (2.0 * h * c**2) / (Lambda**5) / (np.exp(exponent) - 1.0)

    return B_Lambda

# temperature range: 3e0, 3e1, 3e2, ..., 3e8 K
Ts = [3 * 10**i for i in range(0, 9)]

# wavelength Range: 1e0, 1e1, 1e2, ..., 1e9 nm
Lambdas_nm = np.logspace(0, 9, num=1000)``
Lambdas_m  = Lambdas_nm * 1e-9           # convert [nm] to [m]

# Observing Facility: Hubble Space Telescope (Optical ~550 nm)
#hst_wavelength_nm = 550  # nm
#hst_wavelength_cm = hst_wavelength_nm * 1e-7  # Convert nm to cm

# Plot Setup
plt.figure(figsize=(10, 7))

# Colors for different temperatures
colours = plt.cm.viridis(np.linspace(0, 1, len(Ts)))

for i, T in enumerate(Ts):
    Bs_Lambda = Planck_law(Lambdas_m, T)
    plt.plot(Lambdas_nm, Bs_Lambda, label=f"{T} K", color=colours[i])
    
    # Optional: Highlight the peak wavelength using Wien's Displacement Law
    #peak_wavelength_cm = (2.897771955e-1) / T  # Wien's Law: λ_max (cm) = 2.897771955e-1 / T
    #peak_wavelength_nm = peak_wavelength_cm * 1e7
    #plt.scatter(peak_wavelength_nm, planck_wavelength(peak_wavelength_cm, T), color=colors[idx], s=20, marker='o')

# Plot Observing Facility Wavelength
#`plt.axvline(x=hst_wavelength_nm, color='red', linestyle='--', label='HST Optical (~550 nm)')

# Labels and Title
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral Radiance (erg/s/cm²/cm/sr)")
plt.title("Planck Spectrum for Various Temperatures with HST Optical Wavelength")
plt.xscale('log')
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()

# Save the plot as PDF
plt.savefig("Planck_spec_ChuizhengKong.pdf")
plt.show()
