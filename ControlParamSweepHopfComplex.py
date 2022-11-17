import numpy as np
import HopfModelwithcontrol as HopfC
import TurbulenceMeasures as TB
#Control param
control_N = 6
mu = np.linspace(0.94,1,control_N)
Turbulence_measurements_under_control =np.zeros(control_N)
#Simulation parameters
#Model parameters
N=1000; T=1000; delta_x=0.1; beta=2.6; sigma=0.1; K=0.05
#Latent parameters
alpha = np.angle(complex(1,beta)); w=beta+1; D= (sigma/K)*np.sqrt(1+beta**2)
#Coupling function
def G(x):
    return 0.5*np.exp(-np.abs(x))
#Integration parameters
#phi_0 = np.random.uniform(low=0,high=2*np.pi,size=N)
W_0=np.zeros(N)+10**(-7)
num_points = 10000; t_0=0

for c in range(0,control_N):
    print(c+94)
    sol,t = HopfC.SolveComplexHopfModelWithControl(W_0,T,N,w,G,num_points,t_0,delta_x,alpha,sigma,K,beta,mu[c])
    phi = np.angle(sol) - beta*np.log(np.abs(sol))
    bounded_phi = np.mod(phi,2*np.pi)
    LKO_R_Hc,LKO_theta_Hc = TB.CalculateLocalKuramotoOrderParam(bounded_phi,G,delta_x)
    Turbulence_measurements_under_control[c] = TB.CalculateVarianceLKOP(LKO_R_Hc)
    print(Turbulence_measurements_under_control[c])
with open("musweepHopfComplex.txt", "w") as txt_file:
    for line in list(Turbulence_measurements_under_control):
        txt_file.write("".join(str(line)) + "\n")
