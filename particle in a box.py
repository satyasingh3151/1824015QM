import numpy as np
import matplotlib.pyplot as plt
n=100
L=8
x=np.linspace(0,8,100)
dx=x[1]-x[0]
def kinetic_energy(n):
    A=np.zeros((n,n))
    for i in range(n):
        A[i,i]=-2/(dx)**2
        if i>0:
          A[i,i-1]=1/(dx)**2
        if i<(n-i):
          A[i,i+1]=1/(dx)**2
    # print(A)
    kinetic=0.5*A
    return(kinetic)
KE=kinetic_energy(n)
# print(KE)
def potential_energy(V_values):
   V=np.diag(V_values)
   return (V)
PE=potential_energy(np.zeros(n))
# print(PE)
H=PE+KE
E,psi=np.linalg.eigh(H)
# normlize all eigen function
norm=np.sqrt(np.sum(np.abs(psi)**2,axis=0)*dx)
psi_normalized=psi/norm
print(E)
print(psi)
print(np.shape(psi))
N=np.arange(len(E))
plt.plot(N,E,'o-')
plt.xlabel("quantum no.")
plt.ylabel("Energy states")
plt.title("energy vs N")
plt.grid(True)
plt.show()
print(psi_normalized)
plt.plot(x,psi_normalized[:,3])
plt.plot(x,psi[9])
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.title("ground state wave function")
plt.grid("True")
plt.show()


