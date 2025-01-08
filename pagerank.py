import numpy as np
import matplotlib
A1 = np.matrix([0,0,1,1/2],[1/3,0,0,0],[1/3,1/2,0,1/2],[1/3,1/2,0,0])
A2 = np.matrix([0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,1,1,0])

#cette fonction est quasiment impossible à se reproduire
def vecteur_de_score(A,a) :
    S = np.ones(len(A))/(len(A))
    return (1-a)*A + a*S
#meilleur solution
def score_dynamique(A,a,iter_max) :
    n = len(A)
    z = np.ones(1,n)/n
    for i in range(0,iter_max + 1) :
        z1 = np.dot(A,z)
        if np.all(np.fabs(z-z1)<1e-3) :
            break
        z = z1
    return z

#voir l'efficacité des convergences des suites utilisées precedement

def erreur_de_score(A,a) :
    n = len(A)
    z = np.ones(1,n)/n
    for i in range(301) :
        z1 = np.dot(A,z)
        if np.all(np.fabs(z-z1)<1e-3) :
                break
        z = z1


    x = np.random.ranf((n,1)) / np.sum(n)
    erreur = np.zeros(300)
    for k in range(301) :
        erreur[k] = np.sum(np.abs(z-x))
        z = np.dot(A,z)

    return erreur



#erreur1 = erreur_de_score(A1,0.15)
erreur2 = erreur_de_score(A2,0.15)
plt.figure(figsize=(10,6))
#plt.semilogy(erreur1,label="miniwebA")
plt.semilogy(erreur,label="miiniwebB")
plt.xlabel("Itérations")
plt.ylabel("Erreur (échelle logarithmique)")
plt.show

if __name__ == "__main__" :
    print(vecteur_de_score(A1,0.15))
    print("\n")
    print(score_dynamique(A1,0.15,iter_max=300))
    print("\n")
    print(erreur_de_score(A1,0.15))
    print("\n")
