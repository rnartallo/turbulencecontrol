import numpy as np
import HopfModelwithcontrol as HopfC
import TurbulenceMeasures as TB
#Control param
control_N = 100
mu = np.linspace(0,1,control_N)
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
phi_0=np.zeros(N)
num_points = 10000; t_0=0

for c in range(0,control_N):
    print(c+1)
    sol_c,t_c = HopfC.SolveComplexHopfModelWithControl(phi_0,T,N,w,G,num_points,t_0,delta_x,alpha,D,mu[c])
    bounded_sol_c=np.mod(sol_c,2*np.pi)
    LKO_R_Hc,LKO_theta_Hc = TB.CalculateLocalKuramotoOrderParam(bounded_sol_c,G,delta_x)
    Turbulence_measurements_under_control[c] = TB.CalculateVarianceLKOP(LKO_R_Hc)
with open("musweepHopfComplex.txt", "w") as txt_file:
    for line in list(Turbulence_measurements_under_control):
        txt_file.write("".join(str(line)) + "\n")