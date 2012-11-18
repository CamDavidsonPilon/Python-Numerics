"""
Simulate 1-d stochastic differential equations numerically using different schemes


Example:

kappa = 0.3
b = 0.07
sigma = 0.06
gamma = 0.5
delta =0.004
N = 1e6

def drift(x):
    return kappa*( b - x)
    
def diffusion(x):
    return sigma*x**(gamma)

#this is a CIR process

sdeEuler = DiscreteSDE( drift, diffusion, "euler", startPosition = b, delta=delta )
sde.sample( 10, N )
    
    
"""

import scipy.stats as stats
import numpy as np
from time import clock

class DiscreteSDE( object ):
    """
    initalize:
        drift: the drift function, univariate, must accept and return array of same size.
        diffusion the diffusion function, univariate, must accept and return array of same size.
        method: a string in ["euler", "milstein", "second-order" ]
        delta: the time step
        startTime: the starting time of the process
        startPosition: the starting position of the process 
        
    methods:
        sample( t, n): sample the sde n times until time t. Returns a 2d numpy array with time along the columns
    """
    

    def __init__(self, drift, diffusion, method, delta = 0.001, startTime = 0, startPosition =0  ):
        self.drift = drift
        self.diffusion = diffusion
        if method.lower() not in ["euler", "milstein", "second_order" ]:
            raise 
        else:
            self.method = method
        self.delta = delta
        self.startTime = startTime
        self.startPosition = startPosition
            
    def sample(self,t=1, n=1):
        return getattr( self, self.method )(t, n)
        
        
    def euler(self, t, n):
        #initalize
        P,N = self._init(t,n)
        
        for i in xrange(1,int(N)):
            x = P[:, i-1]
            P[:,i] = x + self.drift(x)*self.delta + self.diffusion(x)*np.sqrt(self.delta)*np.random.randn( n )
        
        return P
            
    
    def milstein(self,t,n, h = 0.001):
        
        def diff_prime( u ):
            return (self.diffusion( u + h/2 ) - self.diffusion( u - h/2))/h
        
        
        P, N = self._init(t,n)
        for i in xrange(1,int(N)):
            x = P[:, i-1]
            R = np.random.randn( n )
            P[:,i] = x + self.drift(x)*self.delta + self.diffusion(x)*np.sqrt(self.delta)*R  + \
                                        0.5*diff_prime( x)*self.diffusion(x)*( self.delta*R**2 - self.delta )
        
        return P
        
    def second_order( self, t, n ):
        P, N = self._init(t,n)
        
        
        cov = np.array( [[self.delta, 0.5*self.delta**2],[ 0.5*self.delta**2, self.delta**3/3 ]] )
        mu = np.array( [0,0] )
        for i in xrange(1,int(N)):
            x = P[:, i-1]
            RI = np.random.multivariate_normal( mu, cov, n )
            R = RI[:,0]
            I = RI[:,1]
            
            P[:,i] = x + self.drift(x)*self.delta + self.diffusion(x)*np.sqrt(self.delta)*R  + \
                       (first_derivative( self.drift, x)*self.drift(x) - 0.5*self.diffusion(x)**2*second_derivative( self.drift, x) )*0.5*self.delta**2 + \
                       (first_derivative( self.diffusion, x)*self.drift(x) - 0.5*self.diffusion(x)**2*second_derivative( self.diffusion, x) )*(self.delta*R - I) + \
                       ( self.diffusion(x)*first_derivative(self.drift,x) )*I + \
                       ( self.diffusion(x)*first_derivative(self.diffusion, x) )*(R**2 - self.delta) 
        return P
        
        
        
    def _init(self,t, n ):
        if t < self.startTime:
            raise
        N = np.floor( t / self.delta )
        M = np.zeros( (n, N) )
        M[:,0] = self.startPosition
        return M,N
        
        
def first_derivative( f, x, h = 0.001):
    return ( f(x + h) - f(x-h) )/(2*h)
    
def second_derivative( f, x, h = 0.001):
    return (f(x + h) - 2*f(x) + f(x-h) )/(h**2)

    
    
    
    
if __name__=="__main__":    
 
    kappa = 0.3
    b = 0.07
    sigma = 0.06
    gamma = 0.5
    print "Parameters:"
    print "kappa: %.2f, b: %0.2f, sigma;: %0.2f, gamma: %0.2f"%( kappa, b, sigma, gamma )

    def drift(x):
        return kappa*( b - x)
        
    def diffusion(x):
        return sigma*x**(gamma)
        
    delta =0.004
    sdeEuler = DiscreteSDE( drift, diffusion, "euler", startPosition = b, delta=delta )
    sdeMilstein = DiscreteSDE( drift, diffusion, "milstein", startPosition = b, delta = delta)
    sdeSecondOrder = DiscreteSDE( drift, diffusion, "second_order", startPosition = b, delta = delta)

    
    N = 5000
    print "delta = 0.004"
    
    start = clock()
    eulerAt3 = sdeEuler.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "Euler: q = %.3f s.t. P( X_3 > q ) <= 0.1. Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    
    start = clock()
    eulerAt3 = sdeMilstein.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "Milstein: q = %.3f s.t. P( X_3 > q ) <= 0.1.Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    
    start = clock()
    eulerAt3 = sdeSecondOrder.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "SecondOrder: q = %.3f s.t. P( X_3 > q ) <= 0.1. Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    
    
    print
    delta = 0.1
    print "delta = 0.1"
    start = clock()
    sdeEuler.delta = sdeMilstein.delta = sdeSecondOrder.delta = delta
    
    start = clock()
    eulerAt3 = sdeEuler.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "Euler: q = %.3f s.t. P( X_3 > q ) <= 0.1. Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    
    start = clock()
    eulerAt3 = sdeMilstein.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "Milstein: q = %.3f s.t. P( X_3 > q ) <= 0.1. Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    
    eulerAt3 = sdeSecondOrder.sample( 3, N )[:, -1]
    eulerAt3.sort()
    print "Second Order: q = %.3f s.t. P( X_3 > q ) <= 0.1. Time: %.3f"%(eulerAt3[ np.floor(0.9*N) ], clock() -start)
    

    """
     A bond price is given by:
     P(0,T) = E[ exp( -\int_0^T r_t dt ) ]
     Is the question asking use to compute this integral, which includes the integration? Sure I'll do it.
     
     P(0,T) ~= 1/N * exp( -delta*( \sum r_i ) )
    
    
    """
    
    def bond_price( r_t, delta):
        return np.exp( -delta*(r_t.sum()) )
        
        
    
    def print_partB( discreteSDE, end_time, delta, name ):
        start = clock()
        discreteSDE.delta = delta
        value = stats.nanmean(np.apply_along_axis( lambda u: bond_price(u, delta), 1, discreteSDE.sample( end_time, N )) )
        print "%s: estimates %.4f on %d year bond. Delta: %.4f, Time: %.2f"%(name, value, end_time, delta, clock() - start)
        return
     
    print_partB( sdeEuler, 3, 0.004, "Euler" )
    print_partB( sdeEuler, 3, 0.1, "Euler" )
    print_partB( sdeEuler, 10, 0.004, "Euler" )
    print_partB( sdeEuler, 10, 0.1, "Euler" )
    print
    
    print_partB( sdeMilstein, 3, 0.004, "Milstein" )
    print_partB( sdeMilstein, 3, 0.4, "Milstein" )
    print_partB( sdeMilstein, 10, 0.004, "Milstein" )
    print_partB( sdeMilstein, 10, 0.1, "Milstein" )
    print
    
    print_partB( sdeSecondOrder, 3, 0.004, "Second-Order" )
    print_partB( sdeSecondOrder, 3, 0.1, "Second-Order" )
    print_partB( sdeSecondOrder, 10, 0.004, "Second-Order" )
    print_partB( sdeSecondOrder, 10, 0.1, "Second-Order" )
    
    
    
        