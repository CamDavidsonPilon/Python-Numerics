

"""
Q6.

The best estimate of c* is about 3.05. To find this I used a gamma distribution to estimate the 
integral for values of c between 0 and q and plotted the results. With this optimal 
value of c*, the expected value is approximatly equal to

E[ 1_{x > q} ] = 1.139e-06



"""


import q2
import numpy as np
import scipy.stats as stats
Q = 3.7
interval = (0, np.infty)

def target(u, c):
    return (2*u**2)/(u-c)*np.exp( - ( u**2 + 2*c*u - c**2 ) )
    
    
potentialC = np.array( [0.1, 0.5, 1, 2, 3, 3.5] )
potentialCprime = np.linspace( 2.5, 3.55, 20)
estimates = np.zeros_like( potentialCprime)
for i,c in enumerate(potentialCprime):
    #sampling_dist = stats.norm( loc = c, scale= 5 )
    sampling_dist = stats.gamma(1, loc=Q )
    target_c = lambda x: target(x,c)
    mci = q2.MCIntegrator( target_c, interval =interval, b_antithetic = False, sampling_dist = sampling_dist, N=100000, verbose= False )
    estimates[i] = mci.estimate()
    
    


#3.0526315789473681 is about best
c_opt = 3.0526

def rayleigh(u):
    return 2*u*np.exp(-u**2)

    
def target(u):
    return ( u > Q)*rayleigh(u)/rayleigh(u-c_opt)
sampling_dist = stats.rayleigh( loc = c_opt, scale=1./np.sqrt(2))
mci = q2.MCIntegrator( target, interval =interval, b_antithetic = False, sampling_dist = sampling_dist, N=100000, verbose= False )
print mci.estimate()
# estimate: 1.13859704116e-06


