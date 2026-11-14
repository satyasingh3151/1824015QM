
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# 1. Define constants (in eV and Angstrom units)

hbar_c = 1973.0       # eV Angstrom
mc2 = 0.511e6         # eV
e2 = 14.4             # eV Angstrom

# hbar^2 / 2m
k = hbar_c**2 / (2 * mc2)

# 2. Define radial grid

N = 5000
r_min = 0.01
r_max = 40.0

r = np.linspace(r_min, r_max, N)
dr = r[1] - r[0]

# 3. Define Coulomb potential

V = -e2 / r

# 4. Construct Hamiltonian matrix

diagonal = 2*k/dr**2 + V

off_diagonal = np.ones(N-1) * (-k/dr**2)

# 5. Calculate eigenvalues and eigenvectors

E, u = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, 3)
)

# 6. Normalize eigenfunctions

for i in range(4):
    u[:, i] = u[:, i] / np.sqrt(
        np.sum(u[:, i]**2) * dr
    )

# 7. Display energy eigenvalues

print("Hydrogen atom energy eigenvalues")

for i in range(4):
    print(f"State {i+1}: {E[i]:.4f} eV")

print("\nGround state energy =", E[0], "eV")
print("First excited state energy =", E[1], "eV")

# 8. Analytical energy values

print("\nAnalytical energy values")

for n in [1, 2]:
    E_exact = -13.6 / n**2
    print(f"n = {n}, E = {E_exact:.4f} eV")

# 9. Plot reduced radial wavefunctions u(r)

plt.figure(figsize=(9,5))

plt.plot(r, u[:,0], label="Ground state (1s)")
plt.plot(r, u[:,1], label="First excited state (2s)")

plt.xlabel("Radial distance r (Angstrom)")
plt.ylabel("u(r)")
plt.title("Hydrogen Atom: Radial Wavefunctions")

plt.legend()
plt.grid()
plt.xlim(0, 15)
plt.show()

# 10. Plot radial wavefunctions R(r) = u(r)/r

R = u / r[:, np.newaxis]

plt.figure(figsize=(9,5))

plt.plot(r, R[:,0], label="Ground state (1s)")
plt.plot(r, R[:,1], label="First excited state (2s)")

plt.xlabel("Radial distance r (Angstrom)")
plt.ylabel("R(r)")
plt.title("Hydrogen Atom: Radial Wavefunctions")

plt.legend()
plt.grid()
plt.xlim(0, 15)
plt.show()
