import numpy as np
import cmath
from scipy.fftpack import fft, ifft

def ComplexHopfModelWithControl(W,t,p):
    N,G,delta_x,alpha,K,h,beta,w,mu = p
    dW = np.zeros(N,dtype=complex)
    #Form matrices for FFT
    G_dis = np.zeros(2*N-1)
    W_padded = np.zeros(2*N-1,dtype=complex)
    k = np.zeros(N)
    ones = np.zeros(2*N-1)
    time_step=int(t/h)
    G_temp=np.zeros(N)
    for n in range(0,N):
        G_temp[n]=G(n*delta_x)
        ones[n]=1
        W_padded[n]=W[n,time_step]
    G_temp_flipped = np.flip(G_temp)
    G_temp_flipped = G_temp_flipped[:len(G_temp_flipped)-1]
    G_dis =np.concatenate((G_temp_flipped,G_temp))
    temp_k = np.abs(ifft(np.multiply(fft(ones),fft(G_dis))))
    S_w=ifft(np.multiply(fft(W_padded),fft(G_dis)))
    S_w = S_w[int(len(S_w)/2):]
    k = temp_k[int(len(temp_k)/2):]
    phi= np.angle(W[:,time_step]) - beta*np.log(np.abs(W[:,time_step]))
    KOP = CalculateGlobalKuramotoOrderParam(phi)
    for n in range(0,N):
        dW[n] = (1+w*1j)*W[n,time_step]-(1+beta*1j)*(np.abs(W[n,time_step])**2)*W[n,time_step]+K*(1/k[n])*S_w[n] + mu*(KOP - W[n,time_step])
    return dW

def SolveComplexHopfModelWithControl(W_0,T,N,w,G,num_points,t_0,delta_x,alpha,sigma,K,beta,mu):
    t = np.linspace(t_0,T,num_points)
    W = np.zeros((N,num_points),dtype=complex)
    W[:,0]=W_0
    h=(T-t_0)/num_points
    p=[N,G,delta_x,alpha,K,h,beta,w,mu]
    noise=np.random.normal(0,np.sqrt(h),(N,num_points))
    wiener = np.cumsum(noise,axis=1)
    for n in range(0,num_points-1):
        W[:,n+1] = W[:,n]+ h*ComplexHopfModelWithControl(W,t[n],p) + np.sqrt(sigma)*h*(wiener[:,n+1]-wiener[:,n])
    return [W,t]

def CalculateGlobalKuramotoOrderParam(phi):
    N = np.shape(phi)[0]
    Z=0
    for n in range(0,N):
        Z+=(1/N)*cmath.exp(phi[n]*1j)
    return Z