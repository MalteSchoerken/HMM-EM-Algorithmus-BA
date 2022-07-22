# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:54:45 2022

@author: malte
"""

import numpy as np
import matplotlib.pyplot as plt

y = np.array([4,3,2,1,0,0,0,0,1,2,3])
y2 = np.array([0,4,3,2,1,0,0,0,0,1,2,3])
y3 = np.random.randint(5,size=(10))
y4 = [4,3,2,1,0,0,0,0,1,2,3,0,4,3,2,1,0,0,0,0,1,2,3,0,4,3,2,1,0,0,0,0,1,2,3,0,4,3,2,1,0,0,0,0,1,2,3,0,4,3,2,1,0,0,0,0,1,2,3]
b = np.ones((4,5)) * 0.2
pi = np.ones((4,)) * 0.25
P = np.ones((4,4)) * 0.25
P2 = np.array([[0.1,0.1,0.4,0.4],[0,0,0.5,0.5],[0.3,0.4,0.3,0],[0.25,0.25,0.25,0.25]])
b = np.array([[0.1,0.1,0.1,0.3,0.4],[0.1,0.1,0.1,0.5,0.2],[0.3,0.3,0.3,0.05,0.05],[0.2,0.1,0.2,0.25,0.25]])

# berechnet Vorwärtswahrscheinlichkeit
def forward(y,b,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n = len(pi) #Anzahl Zustände
    fwd = np.zeros((n,T))
    
    # Initialisierung
    for i in range(n):
       fwd[i,0] = pi[i] * b[i][y[0]]
    
    # Induktion
    for t in range(T-1):
        for j in range(n):
            for i in range (n):
                fwd[j,t+1] += fwd[i,t] * P[i,j]
            fwd[j,t+1] *= b[j][y[t+1]]
    return fwd

# berechnet Rückwärtswahrscheinlichkeit
def backward(y,b,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n = len(pi) #Anzahl Zustände
    bwd = np.zeros((n,T))
    
    # Initialisierung
    bwd[:,T-1] = np.ones_like(bwd[:,T-1])
    
    # Induktion
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
    
    # berechne p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n,T-1))
    sump1 = np.zeros((T-1,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p1[i,j,t] = bwd[j,t+1]*fwd[i,t]*b[j,y[t+1]]*Pm[i,j]
                sump1[t] += p1[i,j,t]
    
    # berechne p2 = \sum_{t}p(Z_{t}=i, Z_{t+1}=j | y, \theta)
    p2 = np.zeros((n,n))
    sump2 = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p2[i,j] += p1[i,j,t]/sump1[t] 
            sump2[i] += p2[i,j]   
    
    # berechne P
    for i in range(n):
        for j in range(n):
            P[i,j] = p2[i,j]/sump2[i]                     
    return P

# eine Iteration von pi
def updatePi(y,b,pim,P):
    n = len(pim) #Anzahl Zustände
    fwd = forward(y, b, pim, P)
    bwd = backward(y, b, pim, P) 
    pi = np.zeros_like(pim)

    # berechne p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n))
    sump1 = np.zeros((n,))
    for i in range(n):
        for j in range(n):
            p1[i,j] = bwd[j,1]*fwd[i,0]*b[j,y[1]]*P[i,j]
            sump1 += p1[i,j]
            
    # berechne p2 = p(Z_{t}=i, Z_{t+1}=j | y, \theta) 
    p2 = p1/sump1      
    
    # berechne pi
    for i in range(n):
        for j in range(n):
            pi[i] += p2[i,j]  
    return pi

def updateB(y,bm,pi,P):
    T = len(y) #Anzahl Beobachtungen
    n,m = np.shape(bm) #Anzahl Zustände, Ausgabesymbole
    fwd = forward(y, bm, pi, P)
    bwd = backward(y, bm, pi, P) 
    b = np.zeros_like(bm)
    
    # berechne p1 = p(Z_{t}=i, Z_{t+1}=j, y | \theta)
    p1 = np.zeros((n,n,T-1))
    sump1 = np.zeros((T-1,))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p1[i,j,t] = bwd[j,t+1]*fwd[i,t]*bm[j,y[t+1]]*P[i,j]
                sump1[t] += p1[i,j,t]
    
    # berechne p2 = p(Z_{t}=i, Z_{t+1}=j | y, \theta) und p3 = p(Z_{t}=i | y, \theta)
    p2 = np.zeros((n,n,T))
    p3 = np.zeros((n,T))
    for i in range(n):
        for j in range(n):
            for t in range(T-1):
                p2[i,j] = p1[i,j,t]/sump1[t] 
                p3[i,t] += p2[i,j,t] 
                
    #berechne p4 = \sum_t (p3 * 1{y_t=v})           
    p4 = np.zeros((n,m))
    sump4 = np.zeros((n,))
    for i in range(n):
        for j in range(m):
            for t in range(T):
                if (y[t] == j):
                    p4[i,j] += p3[i,t]
            sump4[i] += p4[i,j]
    
    # berechne b
    for i in range(n):
        for j in range(m):
            b[i,j] = p4[i,j]/sump4[i]                     
    return b

# param: y Beobachtungen, n Zustände, m Ausgabesymbole
def EM(y,n,m,maxiter=100):
    # zufällige Initialiesierung
    P = np.random.random((n,n)) 
    pi = np.random.random((n,)) 
    b = np.random.random((n,m)) 
    # Normierung
    for i in range(n):
        P[i,:] = P[i,:]/ np.sum(P[i,:])
        b[i,:] = b[i,:]/ np.sum(b[i,:])
    pi = pi/np.sum(pi)
    
    # Iteration
    likelihood = np.zeros((100,))
    for i in range(maxiter):
        #print(P)
        #print(pi)
        #print(b)
        likelihood[i] = np.sum(forward(y, b, pi, P)[:,len(y)-1])
        P0 = np.copy(P)
        pi0 = np.copy(pi)
        b0 = np.copy(b)
        P = updateP(y, b0, pi0, P0)
        pi = updatePi(y, b0, pi0, P0)
        b = updateB(y, b0, pi0, P0)
        if (np.linalg.norm(P0-P) + np.linalg.norm(b0-b) + np.linalg.norm(pi0-pi) < 10e-8):
            break
    print(i)
    return np.around(P,5),np.around(pi,5),np.around(b,5), likelihood[likelihood != 0]


for i in range(10):    
    P,pi,b,l = EM(y,4,5)
    plt.semilogy(range(len(l)), l, 'b-')
    print("P = ", P)
    print("pi = ", pi)
    print("b = ", b)
    print("l = ", l)
    #print("likelihood = ", np.sum(forward(y, b, pi, P)[:,len(y)-1]))
plt.show()


for i in range(10):    
    P,pi,b,l = EM(y4,4,5)
    plt.plot(range(len(l)), l, 'b-')
    print("P = ", P)
    print("pi = ", pi)
    print("b = ", b)
    print("l = ", l)
print("y = ", y4)    
plt.show()











