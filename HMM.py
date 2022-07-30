# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:54:45 2022

@author: malte
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# Berechnung der Vorwärtswahrscheinlichkeit
def forward(y,b,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n = len(pi) #Anzahl Zustände
    fwd = np.zeros((n,T))
    
    # Initialisierung
    for i in range(n):
       fwd[i,0] = pi[i] * b[i][y[0]]
    
    # Rekursion
    for t in range(T-1):
        for j in range(n):
            for i in range (n):
                fwd[j,t+1] += fwd[i,t] * P[i,j]
            fwd[j,t+1] *= b[j][y[t+1]]
    return fwd

# Berechnung der Rückwärtswahrscheinlichkeit
def backward(y,b,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n = len(pi) #Anzahl Zustände
    bwd = np.zeros((n,T))
    
    # Initialisierung
    bwd[:,T-1] = np.ones_like(bwd[:,T-1])
    
    # Rekursion
    for t in reversed(range(T-1)):
        for i in range(n):
            for j in range (n):
                bwd[i,t] += bwd[j,t+1] * P[i,j] * b[j][y[t+1]]
    return bwd

# Eine Iteration der Übergangsmatrix P
def updateP(y,b,pi,Pm):
    T = len(y) #Anzahl Beobachtungen
    n = len(pi) #Anzahl Zustände
    fwd = forward(y, b, pi, Pm)
    bwd = backward(y, b, pi, Pm) 
    P = np.zeros_like(Pm)
    
    # Berechnung von p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n,T-1))
    sump1 = np.zeros((T-1,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p1[i,j,t] = bwd[j,t+1]*fwd[i,t]*b[j,y[t+1]]*Pm[i,j]
                sump1[t] += p1[i,j,t]
    
    # Berechnung von p2 = \sum_{t}p(Z_{t}=i, Z_{t+1}=j | y, \theta)
    p2 = np.zeros((n,n))
    sump2 = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p2[i,j] += p1[i,j,t]/sump1[t] 
            sump2[i] += p2[i,j]   
    
    # Berechnung von P
    for i in range(n):
        for j in range(n):
            P[i,j] = p2[i,j]/sump2[i]                     
    return P

# Eine Iteration der Startwahrscheinlichkeiten pi
def updatePi(y,b,pim,P):
    n = len(pim) #Anzahl Zustände
    fwd = forward(y, b, pim, P)
    bwd = backward(y, b, pim, P) 
    pi = np.zeros_like(pim)

    # Berechnung von p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n))
    sump1 = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            p1[i,j] = bwd[j,1]*fwd[i,0]*b[j,y[1]]*P[i,j]
            sump1 += p1[i,j]
            
    # Berechnung von p2 = p(Z_{t}=i, Z_{t+1}=j | y, \theta) 
    p2 = p1/sump1      
    
    # Berechnung von pi
    for i in range(n):
        for j in range(n):
            pi[i] += p2[i,j]  
    return pi

# Eine Iteration der Emissionswahrscheinlichkeiten b
def updateB(y,bm,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n,m = np.shape(bm) #Anzahl Zustände, Ausgabesymbole
    fwd = forward(y, bm, pi, P)
    bwd = backward(y, bm, pi, P) 
    b = np.zeros_like(bm)
    
    # Berechnung von p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n,T-1))
    sump1 = np.zeros((T-1,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p1[i,j,t] = bwd[j,t+1]*fwd[i,t]*bm[j,y[t+1]]*P[i,j]
                sump1[t] += p1[i,j,t]
    
    # Berechnung von p2 = p(Z_{t}=i, Z_{t+1}=j | y, \theta) und p3 = p(Z_{t}=i | y, \theta)
    p2 = np.zeros((n,n,T))
    p3 = np.zeros((n,T))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p2[i,j] = p1[i,j,t]/sump1[t] 
                p3[i,t] += p2[i,j,t] 
                
    #Berechnung von p4 = \sum_t (p3 * 1{y_t=v})           
    p4 = np.zeros((n,m))
    sump4 = np.zeros((n,))
    for i in range(n):
        for j in range(m):
            for t in range(T):
                if (y[t] == j):
                    p4[i,j] += p3[i,t]
            sump4[i] += p4[i,j]
    
    # Berechnung von b
    for i in range(n):
        for j in range(m):
            b[i,j] = p4[i,j]/sump4[i]                     
    return b


# Der EM-Algorithmus für HMMs
# Parameter: y Beobachtungen, n Zustände, m Ausgabesymbole, maxiter maximale Iterationen, epsilon Wert für die Abbruchbedingung
def EM(y,n,m,maxiter=100, epsilon = 10e-12):
    # zufällige Initialisierung
    P = np.random.random((n,n)) 
    pi = np.random.random((n,)) 
    b = np.random.random((n,m)) 
    # Normierung
    for i in range(n):
        P[i,:] = P[i,:]/ np.sum(P[i,:])
        b[i,:] = b[i,:]/ np.sum(b[i,:])
    pi = pi/np.sum(pi)
    
    # Iteration
    likelihood = np.zeros((maxiter,))
    for i in range(maxiter):
        likelihood[i] = np.sum(forward(y, b, pi, P)[:,len(y)-1])
        P0 = np.copy(P)
        pi0 = np.copy(pi)
        b0 = np.copy(b)
        P = updateP(y, b0, pi0, P0)
        pi = updatePi(y, b0, pi0, P0)
        b = updateB(y, b0, pi0, P0)
        """ alternative Abbruch Bedingung
        if (np.log(likelihood[i])-np.log(likelihood[i-1]) <= epsilon):
            break
        """ 
        if (np.linalg.norm(P0-P) + np.linalg.norm(b0-b) + np.linalg.norm(pi0-pi) < epsilon):
            break
    return P,pi,b, likelihood[likelihood != 0]

# generiert T Zustände und Beobachtungen für gegebenes HMM
# Parameter: P Übergangsmatrix, b Emissionswahrscheinlichkeiten, pi Startwahrscheinlichkeiten, T Iterationen
def sampleHMM(P,b,pi,T):
    n,m = np.shape(b) #Anzahl Zustände, Ausgabesymbole
    statespace = np.arange(0,n)
    observationspace = np.arange(0,m)
    states = np.zeros((T,))
    observations = np.zeros((T,))
    states[0] = np.random.choice(statespace,p = pi)
    observations[0] = np.random.choice(observationspace,p = b[math.floor(states[0]),:])
    for i in range(1,T):
        states[i] = np.random.choice(statespace,p = P[math.floor(states[i-1]),:])
        observations[i] = np.random.choice(observationspace,p = b[math.floor(states[i]),:])
    return observations, states
    

#%% Test 1: 2 Zustände, 2 Ausgaben, 12 Beobachtungen

np.random.seed(0)

y2 = np.array([0,1,1,1,1,0,0,0,0,1,1,1])

for i in range(10):    
    P,pi,b,l = EM(y2,2,2)
    plt.plot(range(1,len(l)), np.log(l[1:]), 'b-')
    print("P = ", np.around(P,5))
    print("pi = ", np.around(pi,5))
    print("b = ", np.around(b,5))
 
plt.title("EM-Algorithm, 2 States, 12 Observations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()


#%% Test 2: 4 Zustände, 5 Ausgaben, 12 Beobachtungen

np.random.seed(1)

y = np.array([4,3,2,1,0,0,0,0,1,2,3])

for i in range(10):    
    P,pi,b,l = EM(y,4,5)
    plt.plot(range(1,len(l)), np.log(l[1:]), 'b-')
    print("P = ", np.around(P,5))
    print("pi = ", np.around(pi,5))
    print("b = ", np.around(b,5))

plt.title("EM-Algorithm, 4 States, 12 Observations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()

#%% Test 3: 4 Zustände, 5 Ausgaben, 30 Beobachtungen

np.random.seed(1)

y = np.array([4,3,2,1,0,0,0,0,1,2,3,0,1,2,4,4,4,1,1,2,0,1,2,0,0,0,1,2,1,4])

for i in range(10):    
    P,pi,b,l = EM(y,4,5)
    plt.plot(range(1,len(l)), np.log(l[1:]), 'b-')
    print("P = ", np.around(P,5))
    print("pi = ", np.around(pi,5))
    print("b = ", np.around(b,5))
  
plt.title("EM-Algorithm, 4 States, 30 Observations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()
print(len(y))

#%% Test 4: 50 Samples, 4 Zustände, 5 Ausgaben

np.random.seed(1)

pi = np.ones((4,)) * 0.25
b = np.ones((4,5)) * 0.2
P = np.ones((4,4)) * 0.25

y_sample,z_sample = sampleHMM(P, b, pi, 50)
for i in range(10):    
    P,pi,b,l = EM(y_sample.astype(int),4,5)
    plt.plot(range(1,len(l)), np.log(l[1:]), 'b-')
    print("P = ", np.around(P,5))
    print("pi = ", np.around(pi,5))
    print("b = ", np.around(b,5))

plt.title("EM-Algorithm, 4 States, 50 Observations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()

#%% Test 5: 200 Samples, 2 Zustände, 2 Ausgaben

np.random.seed(1)

pi2 = np.array([0.9,0.1])
P2 = np.array([[0.9,0.1],[1,0]])
b2 = np.array([[0.8,0.2],[0.1,0.9]])

y_sample,z_sample = sampleHMM(P2, b2, pi2, 200)
for i in range(10):    
    P,pi,b,l = EM(y_sample.astype(int),2,2)
    plt.plot(range(1,len(l)), np.log(l[1:]), 'b-')
    print("P = ", np.around(P,5))
    print("pi = ", np.around(pi,5))
    print("b = ", np.around(b,5))

plt.title("EM-Algorithm, 2 States, 200 Observations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()



