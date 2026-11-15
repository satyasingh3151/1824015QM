
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Constants
hbar = 1
m = 1
e2 = 1

k = hbar**2/(2*m)

# Radial grid
N = 3000
r_max = 50

dr = r_max/(N+1)
r = np.arange(1, N+1)*dr

# Screening lengths
a_values = [3, 5, 7]

energies = []
wavefunctions = []

# FDM calculation
for a in a_values:

    # Screened Coulomb potential
    V = -(e2/r)*np.exp(-r/a)

    # Hamiltonian matrix
    diagonal = 2*k/dr**2 + V
    off_diagonal = np.full(N-1, -k/dr**2)

    # Find ground state energy
    E, U = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select='i',
        select_range=(0, 0)
    )

    # Eigenvalue
    energies.append(E[0])

    # Eigenfunction
    u = U[:, 0]

    # Normalization
    u = u/np.sqrt(np.sum(u**2)*dr)

    wavefunctions.append(u)

# Print energy
print("Ground State Energy")

for i in range(3):
    print("a =", a_values[i],
          "Energy =", energies[i])

# Plot potential
plt.figure(figsize=(8,5))

for a in a_values:
    V = -np.exp(-r/a)/r
    plt.plot(r, V, label=f"a={a}")

plt.xlabel("r")
plt.ylabel("V(r)")
plt.title("Screened Coulomb Potential")
plt.legend()
plt.grid()
plt.show()

# Plot normalized wavefunctions
plt.figure(figsize=(8,5))

for i in range(3):
    plt.plot(r, wavefunctions[i],
             label=f"a={a_values[i]}")

plt.xlabel("r")
plt.ylabel("u(r)")
plt.title("Ground State Wavefunctions")
plt.legend()
plt.grid()
plt.show()
plt.xlabel("r")
plt.ylabel("R(r)")
plt.title("Radial Wavefunctions")
plt.legend()
plt.grid()
plt.show()

