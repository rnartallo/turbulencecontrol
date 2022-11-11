import numpy as np
import cmath
from scipy.fftpack import fft, ifft

def HopfPhaseModel(phi,t,p):
    N,w,G,delta_x,alpha,D,h = p
    dOsc = np.zeros(N)
    #Form matrices for FFT
    G_dis = np.zeros(2*N-1)
    E_dis = np.zeros(2*N-1,dtype=complex)
    C_dis = np.zeros(N,dtype=complex)
    R_dis = np.zeros(N)
    k = np.zeros(N)
    ones = np.zeros(2*N-1)
    Theta_dis = np.zeros(N)
    time_step=int(t/h)
    G_temp=np.zeros(N)
    for n in range(0,N):
        G_temp[n]=G(n*delta_x)
        E_dis[n]=cmath.exp(phi[n,time_step]*1j)
        ones[n]=1
    G_temp_flipped = np.flip(G_temp)
    G_temp_flipped = G_temp_flipped[:len(G_temp_flipped)-1]
    G_dis =np.concatenate((G_temp_flipped,G_temp))
    temp_k = np.abs(ifft(np.multiply(fft(ones),fft(G_dis))))
    temp_C=ifft(np.multiply(fft(E_dis),fft(G_dis)))
    C_dis = temp_C[int(len(temp_C)/2):]
    k = temp_k[int(len(temp_k)/2):]
    R_dis = np.abs(np.multiply(1/k,C_dis))
    Theta_dis = np.angle(np.multiply(1/k,C_dis))
    for n in range(0,N):
        dOsc[n] = w-R_dis[n]*np.sin(phi[n,time_step]-Theta_dis[n]-alpha)
    return dOsc

def SolvePhaseHopfModel(phi_0,T,N,w,G,num_points,t_0,delta_x,alpha,D):
    t = np.linspace(t_0,T,num_points)
    phi = np.zeros((N,num_points))
    phi[:,0]=phi_0
    h=(T-t_0)/num_points
    p=[N,w,G,delta_x,alpha,D,h]
    noise=np.random.normal(0,np.sqrt(h),(N,num_points))
    wiener = np.cumsum(noise,axis=1)
    for n in range(0,num_points-1):
        phi[:,n+1] = phi[:,n]+ h*HopfPhaseModel(phi,t[n],p) + np.sqrt(D)*h*(wiener[:,n+1]-wiener[:,n])
    return [phi,t]