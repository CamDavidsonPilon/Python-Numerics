

"""
I simulated 10000 variables with CDF F(y) = 1/3(y**5 + y**2 + y) using a acceptance rejection scheme and the inversion method. While both
where fast, AR was much faster than the inversion method, even using a poor sampling scheme. The sampler I used for the AR method 
a M*Uniform distribution, where M = max_y 1/3( 5*y**4 + 2*y + 1). This bounded the pdf of Y. My output is below:

>> Testing AR Method.
>> Generate 10000 variables:
>> Mean: 0.669, time taken: 0.13 seconds

>> Testing Inverse Method.
>> Generate 10000 variables:
>> Mean: 0.669, time taken: 32.76 seconds

"""




import numpy as np
import time
import scipy.stats as stats
from scipy.optimize import fsolve

class AR_method(object):
    def __init__(self, target_f, sample_g, M):
        """
            M: the constant s.t. sample_g*M >= target_f for all x
            sample_g: a scipy.stats frozen random variable.
            target_f: a 1-d integrable, positive function
        """
        self.target_f = target_f
        self.sample_g = sample_g
        self.uniform = stats.uniform
        self.M = M
        
    def generate(self,n=1):
        
        rv = np.zeros( n) 
        i=0
        #recursivly call this.
        while i<n:
            g = self.sample_g.rvs()
            if self.uniform.rvs() < self.target_f(g)/( M*self.sample_g.pdf(g) ):
                rv[i] = g
                i+=1
        return rv
        
        
    def generateII( self, n=1):
        "quicker, uses recursion"
        if n==0:
            return np.array([])
        else:
            #generate n new points
            g = self.sample_g.rvs(size=n)
            
            u = self.uniform.rvs(size=n)
            r = self.target_f(g)/( M*self.sample_g.pdf(g) )
            samples = g[ u < r ]
            
            return np.concatenate( [samples, self.generateII( n = n - samples.shape[0] ) ] )
    
    def generateIII( self, n=1):
        "quickest"
        samples = np.array( [] )
        i = 0
        while i < n:
            g = self.sample_g.rvs(size=n)
            u = self.uniform.rvs(size=n)
            r = self.target_f(g)/( M*self.sample_g.pdf(g) )
            new_samples = g[ u < r ] 
            samples = np.concatenate( [samples, new_samples] )
            i += new_samples.shape[0]
        return samples
        
class Inversion_method(object):
    
    def __init__(self, target_F):
        self.target_f = target_F
        self.uniform = stats.uniform()
        
        
    def generate( self, n=1):
        n = int(n)
        rv = np.zeros(n)
        for i in range(n):
            u = self.uniform.rvs()
            rv[i] = fsolve( lambda y: self.target_f(y) - u, 0.5)
        
        return rv
    
    def generateII(self, n=1):
        n = int(n)
        try:
            return fsolve( lambda y:self.target_f(y) - self.uniform.rvs(size=n), 0.5*np.ones(n) )
        except MemoryError:
            i = 0
            sample = np.array([])
            while i<n:
                d = min(n-i, 750)
                u = self.uniform.rvs(size=d)
                sample = np.concatenate( [sample, fsolve( lambda y:self.target_f(y) - u, 0.5*np.ones(d) )] )
                i+=d
            return sample
        
        
        
if __name__=="__main__":
    F = lambda x: (x+x**2 + x**(5))/3.
    f = lambda x: (1 + 2*x + 5*x**(4))/3.
    N = 1e4
    print "Testing AR Method."
    print "Generate %d variables:"%N
    g = stats.uniform()
    M = f(1)
    ar = AR_method( target_f = f, sample_g = g, M = M)

    start = time.clock()
    ar_test = ar.generateIII( N )
    print "Mean: %.3f, time taken: %.2f seconds"%(ar_test.mean(), time.clock() - start )

    print
    print "Testing Inverse Method."
    print "Generate %d variables:"%N
    iv = Inversion_method( target_F = F)

    start = time.clock()
    iv_test = iv.generateII( N )
    print "Mean: %.3f, time taken: %.2f seconds"%(iv_test.mean(), time.clock() - start )



                
        