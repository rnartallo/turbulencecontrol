import numpy as np
import cmath
from scipy.fftpack import fft, ifft

def CalculateLocalKuramotoOrderParam(phi,G,delta_x):
    N,T = np.shape(phi)
    G_dis = np.zeros(2*N-1)
    k = np.zeros(N)
    ones = np.zeros(2*N-1)
    E_dis = np.zeros(2*N-1,dtype=complex)
    C_dis = np.zeros(N,dtype=complex)
    R_dis = np.zeros((N,T))
    Theta_dis = np.zeros((N,T))
    G_temp=np.zeros(N)
    for n in range(0,N):
        G_temp[n] = G(n*delta_x)
        ones[n]=1
    G_temp_flipped = np.flip(G_temp)
    G_temp_flipped = G_temp_flipped[:len(G_temp_flipped)-1]
    G_dis =np.concatenate((G_temp_flipped,G_temp))
    for t in range(0,T):
        for n in range(0,N):
            E_dis[n]=cmath.exp(phi[n,t]*1j)
        temp=ifft(np.multiply(fft(E_dis),fft(G_dis)))
        temp_k = np.abs(ifft(np.multiply(fft(ones),fft(G_dis))))
        k = temp_k[int(len(temp_k)/2):]
        C_dis = temp[int(len(temp)/2):]
        R_dis[:,t] = np.abs(np.multiply(1/k,C_dis))
        Theta_dis[:,t] = np.angle(np.multiply(1/k,C_dis))
    return [R_dis,Theta_dis]

def CalculateLocalKOnoFFT(phi,G,delta_x):
    N,T = np.shape(phi)
    R_dis = np.zeros(np.shape(phi))
    k = np.zeros(N)
    for n in range(0,N):
        for j in range(0,N):
            k[n]+= G((n-j)*delta_x)
    Theta_dis = np.zeros(np.shape(phi))
    C_dis=np.zeros(N,dtype=complex)
    G_dis = np.zeros(2*N)
    for n in range(0,N):
        G_dis[n] = G(n*delta_x)
        G_dis[-n] = G_dis[n]
    for t in range(0,T):
        for i in range(0,N):
            for j in range(0,N):
                C_dis[i]+=G_dis[i-j]*cmath.exp(phi[j,t]*1j)
        R_dis[:,t] = np.abs(np.multiply(1/k,C_dis))
        Theta_dis[:,t] = np.angle(np.multiply(1/k,C_dis))
    return [R_dis,Theta_dis]