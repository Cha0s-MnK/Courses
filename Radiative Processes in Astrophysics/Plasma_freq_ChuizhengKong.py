import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 1.602e-19  # Charge of an electron (C)
me = 9.109e-31  # Mass of an electron (kg)
eps0 = 8.854e-12  # Permittivity of free space (F/m)
c = 3e8  # Speed of light (m/s)

# Frequency range of LHAASO telescope (in Hz)
LHAASO_min_freq = 1e8  # 100 MHz
LHAASO_max_freq = 1e9  # 1 GHz

# Define range of electron densities (in m^-3) and magnetic field strengths (in Tesla)
electron_densities = np.logspace(6, 12, 100)  # from 1e6 to 1e12 m^-3
B_fields = np.logspace(-9, -3, 100)  # from 1 nT to 1 mT

# Calculate plasma frequency (in Hz)
# Formula: f_plasma = (1 / 2 * pi) * sqrt((n_e * e^2) / (me * eps0))
plasma_frequencies = np.sqrt((electron_densities * e**2) / (me * eps0)) / (2 * np.pi)

# Calculate Larmor frequency (in Hz)
# Formula: f_Larmor = (e * B) / (2 * pi * me)
larmor_frequencies = (e * B_fields) / (2 * np.pi * me)

# Plotting
plt.figure(figsize=(10, 6))

# Plot plasma frequency vs. electron density
plt.loglog(electron_densities, plasma_frequencies, label='Plasma Frequency', color='blue')

# Plot Larmor frequency vs. magnetic field strength
plt.loglog(B_fields, larmor_frequencies, label='Larmor Frequency', color='green')

# Plot the LHAASO telescope frequency range
plt.fill_betweenx([1e6, 1e12], LHAASO_min_freq, LHAASO_max_freq, color='gray', alpha=0.3, label='LHAASO Frequency Range')

# Labels and title
plt.xlabel('Electron Density (m$^{-3}$) / Magnetic Field Strength (T)')
plt.ylabel('Frequency (Hz)')
plt.title('Plasma and Larmor Frequencies with LHAASO Telescope Range')
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Save plot as a PDF
plt.savefig('plasma_larmor_frequencies.pdf')

# Display the plot
plt.show()
