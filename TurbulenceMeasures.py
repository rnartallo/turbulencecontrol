import numpy as np
import cmath
from scipy.fftpack import fft, ifft

def CalculateLocalKuramotoOrderParam(phi,G,delta_x):
    N,T = np.shape(phi)
    G_dis = np.zeros(N)
    k = np.zeros(N)
    E_dis = np.zeros(N,dtype=complex)
    C_dis = np.zeros(N,dtype=complex)
    R_dis = np.zeros(np.shape(phi))
    Theta_dis = np.zeros(np.shape(phi))
    for n in range(0,N):
        for j in range(0,N):
            k[n]+= G((n-j)*delta_x)
    for t in range(0,T):
        for n in range(0,N):
            G_dis[n]=G(n*delta_x)
            E_dis[n]=cmath.exp(phi[n,t]*1j)
        C_dis=np.multiply(1/k,ifft(np.multiply(fft(G_dis),fft(E_dis))))
        R_dis[:,t] = np.abs(C_dis)
        Theta_dis[:,t] = np.angle(C_dis)
    return [R_dis,Theta_dis]