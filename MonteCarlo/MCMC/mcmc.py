
from __future__ import division

"""

I'll begin with the MCMC object. 
   It is a very general instance of a MCMC. It uses a Gaussian random walk to propose the next step.
The first issue is whether to accept or reject instances that fall outside the unit cube (as copulas 
are only defined here), or more generally, fall out of the support of the target distribution. We bias
the results if we use the acceptance ratio target(x')/target(x_n). This is because by immediatly rejecting 
results that are outside the support, we are using a truncated proposal distribution, and this is not 
symmetric. Thus in the below code, I use the ratio target(x')/target(x_n) * norm_cdf( x_n)/norm_cdf(x'). See
http://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
for a full, and great, explaination. 
   I have dynamic step size that targets a certain acceptance rate (if too many acceptances, likely
not exploring the space very well, vs. too few acceptances mean likely stepping too far. See documentation in the code). 
   
 


"""

import pdb
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy as sp


# Need a way to sample from copula
# Do this using MCMC


class Normal_proposal( object ):
    
    def __init__(self, ):
        self.norm = stats.norm
    
    def rvs(self, loc = 0, scale = 1, size =1):
        return self.norm.rvs( loc = loc, scale = scale, size = size )
    
    def pdf( self, x, given, scale= 1 ):
        return self.norm.pdf( x-given, scale = 1).prod() #assumes independent
    
   


class MCMC(object):
        """
        Implementation of the Metropolis-Hasting algo.
        params:
            target_dist: the target_distribution, what accept a d-dim vector.
            proposal_dist: the proposal dist, an object with the following methods:
                .pdf(x, y, scale): the pdf of scale*X | y, should accept a vector
                .rvs(loc, scale, size) #todo
                
            x_0: a starting location
            burn_in: the number of burn in steps
            dim: the dimension of the densities.
            init_scale: the initial scale to start at. The algorithm uses a simple 
                dynamic scale to target a certain acceptance ratio.
            
        methods:
            next() : generates and returns a random variate from the target_dist
            
            
        """
        def __init__(self, target_dist, 
                           dim = 1, 
                           x_0 = None, 
                           burn_in = 300, 
                           init_scale = 1,
                           proposal_dist = Normal_proposal(),
                           verbose = True):
            self.target_dist = target_dist
            self.x = x_0
            self.burn_in = burn_in
            self.dim = dim
            self.uniform = stats.uniform()
            #self.std = 1
            self.proposals = 0
            self.accepted = 0
            self.proposal_dist = proposal_dist
            self.verbose = verbose
            self.std = init_scale
            self.array_std = self.std*np.ones(1)
            if x_0 == None:
                #initialize array
                self.x = np.zeros(dim)
            self._burn()
            
        def _normcdf(self, x_array):
            return proposal_dist.cdf( x_array).prod()
        
        def _modify_step(self):
            #lets check our acceptance rate, and aim for .234, see http://www.maths.lancs.ac.uk/~sherlocc/Publications/rwm.final.pdf
            opt_rate = .234
            epsilon = 0.05
            rate = self.accepted/self.proposals
            if rate > opt_rate + epsilon: #too many acceptance, spread out more
                self.std *= 1.001
            elif rate < opt_rate  - epsilon:
                self.std /= 1.001
            
            self.array_std = np.append( self.array_std, self.std)
            return
            
        def rvs(self, n=1):
            #generate a new sample
            #An interesting bug: http://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
            
            observations = np.empty( (n,self.dim) )
            for i in range(n):
                accept = False
                tally = 0
                #lets keep a running tally of our acceptance rate.
                while not accept:
                    self.proposals += 1
                    #x_new = self.x +  self.std*np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim))
                    x_new = self.proposal_dist.rvs(self.x, scale = self.std, size = self.dim) #this is 
                    #a = self.target_dist( x_new )/ self.target_dist( self.x) #we use the correct acceptance ratio:
                    #a = self.target_dist( x_new)*self._normcdf( self.x)/ ( self.target_dist( self.x )*self._normcdf( x_new ) )
                    a = self.target_dist(x_new)*self.proposal_dist.pdf(self.x, x_new)/( self.target_dist(self.x)*self.proposal_dist.pdf( x_new, self.x) )
                    #print a
                    #pdb.set_trace()
                    if (a>=1) or ( self.uniform.rvs() < a ):
                        accept = True
                        self.x = x_new
                        self.accepted +=1
                    tally+=1
                    if tally%150==0:
                        print "hmm...I'm not mixing well. I've rejected 150+ samples. Try a restart? Currently at ", self.x
                observations[i] = self.x
            return observations  
  
        def _burn(self):
            if self.verbose:
                print "Burn, Baby, burn. %d times."%self.burn_in
            for i in xrange(self.burn_in):
                self.rvs()
                self._modify_step()

            if self.verbose:
                print "Burn-in complete. Use next() to call new observations."

