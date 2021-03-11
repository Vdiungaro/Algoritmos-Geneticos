import numpy as np
import time
import random
import math
from scipy.spatial import distance
import Functions
import statistics

Popsize = 30
dim = 2
ep = 0.0001
g0 = 100
alfa = 20
t = 1
lb = -5
ub = 5

def f(x,y):
    return ((1-x) ** 2) + (100*(y-(x**2))**2)

def funcapt(Particle):
    a,b = np.split(Particle,dim)
    return np.round(f(a,b),4)

def calculateMass(apt,best,worst):
    m = np.zeros(shape=(Popsize,1),dtype = float)
    M_iner = np.zeros(shape=(Popsize,1),dtype=float)

    for i in range(Popsize):
        m[i] = (apt[i] - apt[worst])/((apt[best] - apt[worst])) + ep
    Msum = np.sum(m)

    for i in range(Popsize):
        M_iner[i] = (m[i])/Msum
    #print(M_iner)
    return M_iner

def calculateForce(g_cte,Masses, X):

    force = np.zeros(shape=(Popsize,dim),dtype=float)

    for i in range(0,Popsize):
        f = None
        for j in range(1,Popsize):
            dividend = float(Masses[i] * Masses[j])
            divisor = distance.euclidean(X[i],X[j]) + np.finfo('float').eps

            if f is None:
                f = g_cte * (dividend/divisor) * (np.subtract(X[j],X[i]))
            else:
                f = f + g_cte * (dividend/divisor) * (np.subtract(X[j],X[i]))
            #print(f)
        force[i] = np.random.uniform(0,1) * f

    return force

def calculateaccelarition(forces,Masses):

    acc = np.zeros(shape=(Popsize,dim),dtype=float)

    for i in range(Popsize):
        acc[i] = forces[i]/Masses[i]

    return acc

def Move(V,X,acc):

    for i in range(Popsize):
        randnum = random.random()
        V[i] = np.add((randnum*V[i]),acc[i])
        # np.clip na velocidade pros limites
        X[i] = np.add(X[i],V[i])
        # provavelmente np.add

    X = np.clip(X,lb,ub)
    V = np.clip(V,-1,1)
    return X,V

## Massa, apt = 1 dim ,  resto = Popsize,dim

def gsa(X,V,max_it):
    best_so_far = []
    t = 0
    obj = None
    best_mean = []
    apt_mean = []
    while t<max_it:
        apt = np.zeros(shape=(Popsize,1))
        for i in range(Popsize):
            apt[i] = funcapt(X[i])
        #print(apt)

        best = np.argmin(apt)
        worst = np.argmax(apt)


        Mass = calculateMass(apt,best,worst)
        #print(Mass)

        g_cte = g0*math.exp(-alfa*(t/max_it))

        sum = np.sum(X)
        sum_nan = np.isnan(sum)

        if sum_nan == True :
            print('Falhou')
            break

        Force = calculateForce(g_cte,Mass,X)
        #print(Force)

        acc = calculateaccelarition(Force,Mass)
        #print(acc)

        X,V = Move(V,X,acc)
        #print(X[1])


        new_apt = np.zeros(shape=(Popsize,1))
        for i in range(Popsize):
            new_apt[i] = funcapt(X[i])

        aux = np.mean(new_apt)
        apt_mean.append(aux)

        best_value = np.amin(new_apt)
        #print(best_value)
        best_so_far.append(best_value)
        best_mean.append(statistics.mean(best_so_far))

        if best_value <= 0.009:
            obj = 1
            break
        t += 1


    return t+1,obj,best_mean,apt_mean


def main():

    start = time.time()

    aux1=np.zeros(shape=(30,1),dtype='float64')
    aux2=np.zeros(shape=(30,1),dtype='float64')

    for i in range(30):
        X = np.random.uniform(lb,ub,size=(Popsize*dim))
        X = np.reshape(X,(Popsize,dim))

        V = np.random.uniform(-1, 1, size=(Popsize * dim))
        V = np.reshape(V, (Popsize, dim))

        iter ,obj, best_mean,apt_mean = gsa(X,V,max_it=500)

        aux1[i] = (statistics.mean(best_mean))
        aux2[i] = (statistics.mean(apt_mean))

        if obj == 1:
            print('Sucesso',i)

    end = time.time()

    #print(aux1,aux2)
    z1 = np.nanmean(aux1)
    z2 = np.nanmean(aux2)
    print(z1,z2)
    Functions.plot_bestmean(best_mean,file_path='Results/Func4/bestmean.pdf',title='F4')
    Functions.plot_aptmean(apt_mean, file_path='Results/Func4/aptmean.pdf',title='F4')

    print((end-start),'segundos')
if __name__ == '__main__':
    main()