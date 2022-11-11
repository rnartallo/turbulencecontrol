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
    Theta_dis = np.zeros(N)
    time_step=int(t/h)
    G_temp=np.zeros(N)
    for n in range(0,N):
        #for j in range(0,N):
            #k[n]+= G((n-j)*delta_x)
        G_temp[n]=G(n*delta_x)
        E_dis[n]=cmath.exp(phi[n,time_step]*1j)
    G_temp_flipped = np.flip(G_temp)
    G_temp_flipped = G_temp_flipped[:len(G_temp_flipped)-1]
    G_dis =np.concatenate((G_temp_flipped,G_temp))
    temp=ifft(np.multiply(fft(E_dis),fft(G_dis)))
    C_dis = temp[int(len(temp)/2):]
    R_dis = np.abs(C_dis)
    Theta_dis = np.angle(C_dis)
    noise=np.random.normal(0,1,N)
    for n in range(0,N):
        dOsc[n] = w-R_dis[n]*np.sin(phi[n,time_step]-Theta_dis[n]-alpha)+np.sqrt(D)*noise[n]
    return dOsc

def SolvePhaseHopfModel(phi_0,T,N,w,G,num_points,t_0,delta_x,alpha,D):
    p=[N,w,G,delta_x,alpha,D,]
    t = np.linspace(t_0,T,num_points)
    phi = np.zeros((N,num_points))
    phi[:,0]=phi_0
    h=(T-t_0)/num_points
    p=[N,w,G,delta_x,alpha,D,h]
    for n in range(0,num_points-1):
        phi[:,n+1] = phi[:,n]+ h*HopfPhaseModel(phi,t[n],p)
    return [phi,t]