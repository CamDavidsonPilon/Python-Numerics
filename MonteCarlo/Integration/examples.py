
import numpy as np
import scipy.stats as stats
import time

from MonteCarloIntegrator import *

"""
Lets estimate the integral

I = \int_0^2 exp(-x**2) dx
  = E_u[ exp(-x**2)/2 ] where u ~ Uni(0,2)


"""


def target(u):
    return np.exp(-u**2)*2

mci = MCIntegrator( target, interval =(0,2), b_antithetic = False, sampling_dist = stats.uniform(0,2), verbose= True )
N = 1e6

start = time.clock()
print "Using %d samples,"%N
print "Non-antithetic: %.5f."%mci.estimate_N(N )
print "Duration: %.3f s."%(time.clock() - start) 

#using anti-thetic

mci = MCIntegrator( target, interval =(0,2), b_antithetic = True, sampling_dist = stats.uniform(0,2), verbose= True )
start = time.clock()
print "Antithetic: %.5f."%mci.estimate_N(N )        
print "Duration: %.3f s."%(time.clock() - start) 



#Using importance sampling

def importance_function(u):
    return (-.5*u + 1)*2
    
    
class Importance(object):
    def __init__(self):
        pass
    
    def rvs(self,n):
        u = stats.uniform(0,1).rvs( n)
        return 2*( 1 - np.sqrt(u) )

sampling_dist = Importance()
mci = MCIntegrator( target, interval = (0,2), b_antithetic = False, sampling_dist = sampling_dist, N=N, verbose= True )
print mci.estimate()


#using control variates

def polynomial_control( u ):
    return -.26*( (1-u**2) - -1.0/3)
    
mci = MCIntegrator( target, interval =(0,2), sampling_dist = stats.uniform(0,2), verbose= True, N=N, control_variates=[polynomial_control] )
start = time.clock()
print "Control Variates: %.5f."%mci.estimate_N(N )        
print "Duration: %.3f s."%(time.clock() - start)