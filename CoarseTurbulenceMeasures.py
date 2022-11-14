import numpy as np
import cmath
from scipy.fftpack import fft, ifft
from itertools import combinations

def CalculateEdgeCenteredMatrix(phi):
    N,T = np.shape(phi)
    indexes = list(range(0,N))
    pairs = list(combinations(indexes,2))
    E = np.zeros((int(N*(N-1)/2),T))
    for n in range(0,int(N*(N-1)/2)):
        i,j = pairs[n]
        for t in range(0,T):
            E[n,t]=np.sqrt((np.cos(phi[i,t])-np.cos(phi[j,t]))^2 + (np.sin(phi[i,t])-np.sin(phi[j,t]))^2)
    return [E,pairs]

def CalculateEdgeMetastability(E):
    return np.var(E)

def Demean(E):
    M,T = np.shape(E)
    E_means = np.mean(E,axis=0)
    E_bar = np.zeros(M,T)
    for m in range(0,M):
        E_bar[m,:]=E[m,:]-E_means[m]
    return(E_bar)


def CalculateEdgeSpaceTimeProb(E,tau_range):
    M,T = np.shape(E)
    C = np.zeros((M,len(tau_range)))
    E_bar = Demean(E)
    for m in range(0,M):
        for tau in tau_range:
            num=0;den1 =0;den2 =0
            for t in range(tau,T):
                num+=E_bar[m,t-tau]*E_bar[m,t]
                den1+=E_bar[m,t-tau]^2
                den2+=E_bar[m,t]^2
            C[m,tau]=(num)/(np.sqrt(den1)*np.sqrt(den2))
    return C
