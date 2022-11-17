import numpy as np
import Modelwithcontrol as model_c
import TurbulenceMeasures as TB

#Control param
control_N = 30
mu = np.linspace(0.7,1,control_N)
Turbulence_measurements_under_control =np.zeros(control_N)
#Simulation parameters
#Model parameters
N=1000; T=1000; delta_x=0.1; beta=2.6; sigma=0.1; K=0.05
#Latent parameters
alpha = np.angle(complex(1,beta)); w=beta+1; D= (sigma/K)*np.sqrt(1+beta**2)
f = open('output1.txt','w')
#Coupling function
def G(x):
    return 0.5*np.exp(-np.abs(x))
#Integration parameters
#phi_0 = np.random.uniform(low=0,high=2*np.pi,size=N)
phi_0=np.zeros(N)
num_points = 10000; t_0=0
for c in range(0,control_N):
    print(c+70)
    sol_c,t_c = model_c.SolvePhaseHopfModelControlled(phi_0,T,N,w,G,num_points,t_0,delta_x,alpha,D,mu[c])
    bounded_sol_c=np.mod(sol_c,2*np.pi)
    LKO_R_c,LKO_theta_c = TB.CalculateLocalKuramotoOrderParam(bounded_sol_c,G,delta_x)
    Turbulence_measurements_under_control[c] = TB.CalculateVarianceLKOP(LKO_R_c)
    print(Turbulence_measurements_under_control[c])
    f.write("".join(str(Turbulence_measurements_under_control[c])) + "\n")
f.close()
