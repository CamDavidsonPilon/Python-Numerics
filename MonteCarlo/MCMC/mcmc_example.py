"""
Use MCMC to sample from some copulas

"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy as sp

from mcmc import *

def gumbel(t, theta = 1):
    #theta in (0, \infty)
    return np.exp( -t**(1./theta) )
    
def inv_gumbel( t, theta=1):
    return -np.log(t)**theta
    
    
def clayton(t, theta=1):
    return (1+theta*t)**(-1./theta)
    
def inv_clayton( t, theta =1):
    return 1.0/theta*( t**(-theta) - 1)
    

    
def arch_copula(u, f= gumbel, f_inv = inv_gumbel, theta = 1 ):
    """
    #u is a numpy array
    """

    if ( (u > 1).sum() + (u <0).sum() )>0:
        return 0
    
    return f( f_inv( u, theta ).sum(), theta )

 
def _pdf(f, u, delta = 0.001 ):
    n = u.shape[0]
    if n==1:
        t= f(u[0]+delta/2) - f(u[0]-delta/2)
        return t
    else:
        f_plus = lambda *x: f( u[0] + delta/2, *x)
        f_minus = lambda *x: f( u[0] - delta/2, *x)
        return _pdf(f_plus, u[1:], delta ) - _pdf(f_minus, u[1:], delta ) 
        
def cdf2pdf( f, u, delta=0.001, kwargs={} ):
    """numerically unstable for large dimensions"""
    def _wrapper(*args):
        u = np.array(args)
        return f(u, **kwargs)
    n = u.shape[0]
    return _pdf( _wrapper, u, delta)/delta**n
    
    
    
mcmc1 = MCMC( lambda u: cdf2pdf( arch_copula, u) , dim = 2, x_0 = np.array( [0.0, 0.0] ) )
mcmc3 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"theta":3}) , dim = 2, x_0 = np.array( [0.0, 0.0] ) )


dataTheta1 = np.concatenate( [mcmc1.next()[:,None].T for i in range(1000)], axis=0)
dataTheta3 = np.concatenate( [mcmc3.next()[:,None].T for i in range(1000)], axis=0)

plt.figure()

plt.subplot(221)
plt.scatter( dataTheta1[:,0], dataTheta1[:,1] )
plt.title(r"1000 values from a Gumbel copula with $\theta$=1")

plt.subplot(222)
plt.scatter( dataTheta3[:,0], dataTheta3[:,1] )
plt.title(r"1000 values from a Gumbel copula with $\theta$=3")



#lets make the exponential
def make_exp( u ):
    return -np.log(u)

plt.subplot(223)
plt.scatter( make_exp( dataTheta1[:,0]) , make_exp( dataTheta1[:,1] ) )
plt.title(r"1000 EXP(1) values from a Gumbel copula with $\theta$=1")


plt.subplot(224)
plt.scatter( make_exp( dataTheta3[:,0]) , make_exp( dataTheta3[:,1] ) )
plt.title(r"1000 EXP(1) values from a Gumbel copula with $\theta$=3")

plt.savefig( "GumbelCopula.pdf")
plt.show()

plt.figure()

mcmc1 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"f":clayton, "f_inv":inv_clayton}  ) , dim = 2, x_0 = np.array( [0.0, 0.0] ) )
mcmc3 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"theta":.2, "f":clayton, "f_inv":inv_clayton}) , dim = 2, x_0 = np.array( [0.0, 0.0] ) )


dataTheta1 = np.concatenate( [mcmc1.next()[:,None].T for i in range(1000)], axis=0)
dataTheta3 = np.concatenate( [mcmc3.next()[:,None].T for i in range(1000)], axis=0)
plt.figure()

plt.subplot(221)
plt.scatter( dataTheta1[:,0], dataTheta1[:,1] )
plt.title(r"1000 values from a Clayton copula with $\theta$=1")

plt.subplot(222)
plt.scatter( dataTheta3[:,0], dataTheta3[:,1] )
plt.title(r"1000 values from a Clayton copula with $\theta$=3")



#lets make the exponential
def make_exp( u ):
    return -np.log(u)

plt.subplot(223)
plt.scatter( make_exp( dataTheta1[:,0]) , make_exp( dataTheta1[:,1] ) )
plt.title(r"1000 EXP(1) values from a Clayton copula with $\theta$=1")


plt.subplot(224)
plt.scatter( make_exp( dataTheta3[:,0]) , make_exp( dataTheta3[:,1] ) )
plt.title(r"1000 EXP(1) values from a Clayton copula with $\theta$=3")
