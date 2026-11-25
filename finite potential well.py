import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
N=500
L=10
x=np.linspace(-L,L,N)
dx=x[1]-x[0]
def kinetic_energy():
    A=np.zeros((N,N))
    for i in range(N):
        A[i,i]=-2/dx**2
        if i>0:
            A[i,i-1]=1/dx**2
        if i<N-1:
            A[i,i+1]=1/dx**2
    kinetic=-0.5*A
    return(kinetic)
KE=kinetic_energy()

V0=10
a=1
V=np.zeros(N)
for i in range(N):
    if np.abs(x[i])>a:
        V[i]=V0
PE=np.diag(V)
H=PE+KE
E,psi=eigh(H)

bound=np.where((E>0)&(E<V0))[0]
E_bound=E[bound]
psi_bound=psi[:,bound]
norm=np.sqrt(np.sum(np.abs(psi_bound)**2,axis=0)*dx)
psi_normalized=psi_bound/norm

probability_density=np.abs(psi_normalized)**2

print("energy of bound states")
for i in range(len(E_bound)):
    print(f"states={i+1}:E={E_bound[i]}")

for i in range(len(E_bound)):
    plt.plot(x,psi_normalized[:,i],label=f"n={i+1}")
plt.xlabel("x")
plt.ylabel("wavefunction")
plt.title("Finite potential well:wave function")
plt.legend()
plt.grid()
plt.show()

for i in range(len(E_bound)):
    plt.plot(x,probability_density[:,i],label=f"n={i+1}")
plt.xlabel("x")
plt.ylabel("probability_density")
plt.title("Finite potential well:probability_density")
plt.legend()
plt.grid()
plt.show()


plt.plot(x,V,label="potential(V(x))")
for i in range(len(E_bound)):
    plt.axhline(E_bound[i], linestyle="--",
                label=f"E{i+1}={E_bound[i]:.3f}")

plt.xlabel("Position x")
plt.ylabel("Energy")
plt.title("Finite Potential Well: Bound Energy Levels")
plt.legend()
plt.grid()
plt.show()