import numpy as np
import cmath
from scipy.fftpack import fft, ifft

def ComplexHopfModelNetwork(W,t,p):
    N,A,K,h,beta,w = p
    dW = np.zeros(N,dtype=complex)
    S_w = np.zeros(N,dtype=complex)
    time_step=int(t/h)
    for n in range(0,N):
        for j in range(0,N):
            S_w[n]+=A[n,j]*W[n,time_step]
    for n in range(0,N):
        dW[n] = (1+w*1j)*W[n,time_step]-(1+beta*1j)*(np.abs(W[n,time_step])**2)*W[n,time_step]+K*S_w[n]
    return dW

def SolveComplexHopfModelNetwork(W_0,T,N,w,A,num_points,t_0,sigma,K,beta):
    t = np.linspace(t_0,T,num_points)
    W = np.zeros((N,num_points),dtype=complex)
    W[:,0]=W_0
    h=(T-t_0)/num_points
    p=[N,A,K,h,beta,w]
    noise=np.random.normal(0,np.sqrt(h),(N,num_points))
    wiener = np.cumsum(noise,axis=1)
    for n in range(0,num_points-1):
        W[:,n+1] = W[:,n]+ h*ComplexHopfModelNetwork(W,t[n],p) + np.sqrt(sigma)*h*(wiener[:,n+1]-wiener[:,n])
    return [W,t]