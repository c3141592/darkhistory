import numpy as np

# Fundamental constants
mp          = 0.938e9                     # proton mass in eV
me          = 510998.9                    # electron mass in eV
hbar        = 6.58211951e-16              # hbar in eV s
c           = 299792458e2                 # speed of light in cm/s
kB          = 8.6173324e-5                # Boltzmann constant in eV/K
alpha       = 1/137.035999139             # fine structure constant
ele         = 1.60217662e-19

# Atomic and optical physics

# Thomson cross section in cm^2
thomson_xsec = 6.652458734e-25
# Stefan-Boltzmann constant in eV^-3 cm^-2 s^2
stefboltz    = np.pi**2 / (60 * (hbar**3) * (c**2))
# 1 rydberg in eV
rydberg      = 13.60569253
# Lyman alpha transition energy in eV
lya_eng      = rydberg*3/4
# Lyman alpha transition frequency in Hz
lya_freq     = lya_eng / (2*np.pi*hbar)
# Hydrogen 2s to 1s decay width in s^-1
width_2s1s    = 8.23
# Bohr radius in cm
bohr_rad     = (hbar*c) / (me*alpha)
# Classical electron radius in cm
ele_rad      = bohr_rad * (alpha**2)
# Electron compton wavelength in cm
ele_compton  = 2*np.pi*hbar*c/me