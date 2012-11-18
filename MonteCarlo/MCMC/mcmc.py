#Q5

"""
I won't be modest. This is some of the god-damn best code I've ever written. I mean, this is 
pretty much as good as it gets. I'll probably write a few blog posts just on this code.

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
   
 

Given a copula, we need to find its pdf. I chose, to establish arbitrary dimensional copulas, to do 
this numerically. I needed to compute the copula differentiated with respect to all of its arguemnts. This
was quite the algorithmic challenge, but I reduced it to a recursive problem that works blazingly fast. This 
felxibility allows us to never have to explicitly find the pdf, which can be difficult even for dimension > 2. 
The differentiation algorithm uses a central difference scheme. Unfortunatly the scheme is unstable for dimensions
greater than 6.



Attached are two figures: one with the Gumbel copula (theta = 1, theta =3 ) and with the Clayton copula (theta=1, theta=3). 
What is interesting with these simulations is we can varying amounts of covariance between the variables as we increase one
or the other. This is very different compared to using a standard covariance matrix, which assume constant covariance thoughout
the support. 



"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy as sp


# Need a way to sample from copula
# Do this using MCMC

class MCMC(object):
        """
        Implementation of the Metropolis-Hasting algo.
        params:
            target_dist: the target_distribution, what accept a d-dim vector.
            proposal_dist: the proposal dist, a scipy.stats frozen instance
            x_0: a starting location
            burn_in: the number of burn in steps
            dim: the dimension of the densities.
            
        methods:
            next() : generates and returns a random variate from the target_dist
            
            
        """
        def __init__(self, target_dist, dim = 1, x_0 = None, burn_in = 100 ):
            self.target_dist = target_dist
            self.x = x_0
            self.burn_in = 100
            self.dim = dim
            self.uniform = stats.uniform()
            self.std = 1
            self.proposals = 0
            self.accepted = 0
            
            if x_0 == None:
                #initialize array
                self.x = np.zeros(dim)
            self._burn()
            
        def _normcdf(self, x_array):
            return stats.norm.cdf( x_array, scale=self.std).prod()
        
        def _modify_step(self):
            #lets check our acceptance rate, and aim for .234, see http://www.maths.lancs.ac.uk/~sherlocc/Publications/rwm.final.pdf
            opt_rate = .234
            rate = float(self.accepted)/self.proposals
            if rate > opt_rate: #too many acceptance, spread out more
                self.std *= 1.02
            elif rate < opt_rate :
                self.std /= 1.02
            return
            
        def next(self):
            #generate a new sample
            #An interesting bug: http://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
            
 
            accept = False
            #lets keep a running tally of our acceptance rate.
            while not accept:
                self.proposals += 1
                x_new = self.x +  self.std*np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim))
                #a = self.target_dist( x_new )/ self.target_dist( self.x) #we use the correct acceptance ratio:
                a = self.target_dist( x_new)*self._normcdf( self.x)/ ( self.target_dist( self.x )*self._normcdf( x_new ) )
                if (a>=1) or ( self.uniform.rvs() < a ):
                    accept = True
                    self.x = x_new
                    self.accepted +=1

            self._modify_step()
            return self.x    
  
        def _burn(self):
            for i in range(self.burn_in):
                print i
                self.next()
            self.proposals = 0
            self.accepted = 0

