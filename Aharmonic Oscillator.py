
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# 1. Constants
hbar = 1
m = 1
k = 1

# Kinetic energy constant
A = hbar**2/(2*m)

# 2. Radial grid
N = 3000
r_max = 10

dr = r_max/(N+1)
r = np.arange(1, N+1)*dr

# 3. Values of b
b_values = [0, 10, 30]

energies = []
wavefunctions = []

# 4. Finite Difference Method
for b in b_values:

    # Anharmonic oscillator potential
    V = 0.5*k*r**2 + (1/3)*b*r**3

    # Hamiltonian matrix
    diagonal = 2*A/dr**2 + V

    off_diagonal = np.full(N-1, -A/dr**2)

    # Eigenvalues and eigenvectors
    E, U = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select='i',
        select_range=(0, 0)
    )

    # Ground state energy
    energies.append(E[0])

    # Ground state wavefunction
    u = U[:, 0]

    # Normalization
    u = u/np.sqrt(np.sum(u**2)*dr)

    wavefunctions.append(u)

# 5. Print energy eigenvalues
print("GROUND STATE ENERGY")

for i in range(len(b_values)):

    print("b =", b_values[i])
    print("Energy =", energies[i])
    print("-------------------")

# 6. Plot potential
plt.figure(figsize=(8,5))

for b in b_values:

    V = 0.5*k*r**2 + (1/3)*b*r**3

    plt.plot(r, V, label="b = "+str(b))

plt.xlabel("r")
plt.ylabel("V(r)")
plt.title("Anharmonic Oscillator Potential")
plt.legend()
plt.grid()
plt.show()

# 7. Plot normalized wavefunctions u(r)
plt.figure(figsize=(8,5))

for i in range(len(b_values)):

    plt.plot(r, wavefunctions[i],
             label="b = "+str(b_values[i]))

plt.xlabel("r")
plt.ylabel("u(r)")
plt.title("Ground State Wavefunctions")
plt.legend()
plt.grid()
plt.show()

# 8. Plot radial wavefunctions R(r)
plt.figure(figsize=(8,5))

for i in range(len(b_values)):

    R = wavefunctions[i]/r

    plt.plot(r, R,
             label="b = "+str(b_values[i]))

plt.xlabel("r")
plt.ylabel("R(r)")
plt.title("Radial Wavefunctions")
plt.legend()
plt.grid()
plt.show()