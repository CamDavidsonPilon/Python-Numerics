
import numpy as np
import scipy.stats as stats
import time

class MCIntegrator( object ):
    """
    target_function: a function that accepts a n-D array, and returns an n-D array.
    interval: the interval of the integration
    b_antithetic: whether to use antithesis variables. Much quicker, but only useful on monotonic target_functions
    sampling_dist: a scipy frozen distribution with support equal to the interval
    N: number of variables to use in the initial estimate.
    control_variates = a list of function that accepts a nD array, and return an nD array
    """
    def __init__(self, target_function, 
                       interval = (0,1), 
                       N = 10000, 
                       b_antithetic = False, 
                       sampling_dist = stats.uniform(), 
                       verbose=False,
                       control_variates = []):
        self.target_function = target_function
        self.min_interval, self.max_interval = interval
        self.N_ = N
        self.N = 0
        self.sampling_dist = sampling_dist
        self.value =0
        self.b_antithetic = b_antithetic
        self.verbose = verbose
        self.control_variates = control_variates
   
    def estimate_N(self, N ):
        self.N += N
        return self._estimate(N)
        
        
        
    def _estimate(self, N):
        
        #generate N values from sampling_dist
        if not self.b_antithetic:
            U = self.sampling_dist.rvs(N)
            Y = self.target_function( U )
            for func in self.control_variates:
                X =  func(U)
                Y += X
                
            if self.verbose:
                print Y.var()
            self.value +=  Y.sum()
        else:
            U_ = self.sampling_dist.rvs(N/2)
            antiU_ = self.min_interval + (self.max_interval - U_ )
            Y =  (self.target_function( U_ ) + self.target_function( antiU_ ) )
            if self.verbose:
                print Y.var()
            self.value +=Y.sum()
        return self.value / self.N 
        
    def estimate(self):
        self.N += self.N_
        return self._estimate(self.N_)
        


if __name__ == "__main__":
    #Some examples:
    
    
    def target(u):
        return np.exp(-u**2)*2
    
    mci = MCIntegrator( target, interval =(0,2), b_antithetic = False, sampling_dist = stats.uniform(0,2), verbose= True )
    N = 1e6
    
    start = time.clock()
    print "Using %d samples,"%N
    print "Non-antithetic: %.5f."%mci.estimate_N(N )
    print "Duration: %.3f s."%(time.clock() - start) 
    print
    mci = MCIntegrator( target, interval =(0,2), b_antithetic = True, sampling_dist = stats.uniform(0,2), verbose= True )
    start = time.clock()
    print "Antithetic: %.5f."%mci.estimate_N(N )        
    print "Duration: %.3f s."%(time.clock() - start) 
    print
    
    """
    Using 1000000 samples,
    0.474815598284
    Non-antithetic: 0.88140.
    Duration: 0.382 s.

    0.0417625416316
    Antithetic: 0.88216.
    Duration: 0.303 s.
    """
    
    
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
    mci = MCIntegrator( target, interval = (0,2), b_antithetic = False, sampling_dist = sampling_dist, N=100000, verbose= True )
    print mci.estimate()
    
    
    #using control variates

    def polynomial_control( u ):
        return -.26*( (1-u**2) - -1.0/3)
        
    mci = MCIntegrator( target, interval =(0,2), sampling_dist = stats.uniform(0,2), verbose= True, control_variates=[polynomial_control] )
    start = time.clock()
    print "Control Variates: %.5f."%mci.estimate_N(N )        
    print "Duration: %.3f s."%(time.clock() - start)

    
    