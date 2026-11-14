
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
N = 500
x_min = -8
x_max = 8
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]
# 3. Define harmonic potential
V = 0.5 *x**2
# 4. Construct kinetic energy matrix
main_diag = np.ones(N) * 2
off_diag = np.ones(N-1) * (-1)
T = (1/ (2*dx**2)) * (
    np.diag(main_diag)
    + np.diag(off_diag, 1)
    + np.diag(off_diag, -1)
)
# 5. Construct potential energy matrix
U = np.diag(V)
# 6. Hamiltonian matrix
H = T + U
# 7. Find eigenvalues and eigenvectors
energy, wavefunctions = eigh(H)
# 8. Plot potential and wavefunctions
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Harmonic Potential")
for n in range(4):
    psi = wavefunctions[:, n]
    # Scale wavefunction for visualization
    plt.plot(x, energy[n] + 0.5*psi/np.max(np.abs(psi)),
             label=f"n={n}, E={energy[n]:.4f}")
plt.xlabel("Position (x)")
plt.ylabel("Energy")
plt.title("Quantum Harmonic Oscillator using FDM")
plt.legend()
plt.grid()
plt.show()
# 9. Print first four energy eigenvalues
print("Energy eigenvalues:")
for n in range(4):
    print(f"n = {n}, Energy = {energy[n]:.6f}")
