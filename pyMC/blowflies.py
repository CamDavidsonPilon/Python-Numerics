"""
See

BAYESIAN INFERENCE AND MARKOV CHAIN MONTE CARLO BY EXAMPLE
GEOFFK. NICHOLLS

"""
import numpy as np
import pymc as mc
import pandas as pd

#observations
data = pd.read_csv("blowfly97I.csv")
yt = data["total"].value

N = t.shape[0]

r =  mc.Exponential( "r", beta = 1.0 )
b = mc.Exponential( "b", beta = 1000.0 )
lambduh = mc.Exponential( "lambdu", beta = 1.0/1000 )
n_0 = mc.Poisson( "n_0", mu=lambduh)


@mc.deterministic
def n_t( n_0=n_0, r=r, b=b, N=N):
    n = np.empty( N, dtype=object)
    n[0] = n_0
    for i in range( 1, N):
        n[i] =  (r*n[i-1])/( 1.0 + b**4*n[i-1]**4 )
    return n

y = np.empty( N, dtype=object)
for i in range(0, N):
    y[i] = mc.Poisson( "y_%i"%i, mu = n_t[i], observed= True, value = yt[i] )
    
model = mc.Model( {"yt":yt, "nt":n_t, "b":b, "r":r, "n_0":n_0})
mcmc=  mc.MCMC(model)

mcmc.sample( 30000, 15000)

    

    



