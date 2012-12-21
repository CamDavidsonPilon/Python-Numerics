"""Some copulas and helpers for copulas"""

from __future__ import division
import numpy as np
import scipy.stats as stats
import scipy as sp


def gumbel(t, theta = 1):
    #theta in (0, \infty)
    return np.exp( -t**(1./theta) )
    
def inv_gumbel( t, theta=1):
    return (-np.log(t) )**theta
    
    
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
    
    
    
class Copula_Proposal( object ):
    def __init__(self):
        self.norm = stats.norm
        
    def rvs(self, loc, scale, size=1):
        return self.norm.rvs( loc = loc, scale= scale, size = size)
        
    def pdf( self, x, given, scale = 1.0):
        """
        http://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
        """
        return self.norm.cdf( x/scale ).prod()
        
       
        
        
