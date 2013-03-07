##Bayesian Bandit in python

import scipy.stats as stats
import numpy as np



class BayesianBandit( object ):


    def __init__( self, prior_alpha = 1, prior_beta = 1 ):
        self.prior_alpha = 1
        self.prior_beta = 1
        self.betad = stats.beta
        
    
    
    def fit(self, bandits, trials = 10 ):
        """
        Bandits is an object that can be called like bandits.pull(choice) and returns a 0 or 1.
        
        
        """
        n_bandits = len( bandits )
        self.n_pulls = np.zeros( n_bandits )
        self.n_successes = np.zeros( n_bandits )
        self.prior_distibutions = np.array( [self.prior_alpha, self.prior_beta])*np.ones( (n_bandits, 2 ) )

        for i in xrange(trials):
        
            choice = np.argmax( self.betad.rvs( self.prior_distibutions[:,0] + self.n_successes,
                                                self.prior_distibutions[:,1] + self.n_pulls - self.n_successes ) )
            outcome = bandits.pull(choice)
            self.n_pulls[choice] += 1
            self.n_successes[choice] += outcome
        
        self.posterior_alpha =  self.prior_distibutions[:,0] + self.n_successes
        self.posterior_beta = self.prior_distibutions[:,1] + self.n_pulls - self.n_successes
        return
        
    def predict(self, n=1):
        choices = np.zeros( n ) 
        for i in range(n):
        
        
            choice = np.argmax( self.betad.rvs( self.prior_distibutions[:,0] + self.n_successes,
                                                self.prior_distibutions[:,1] + self.n_pulls - self.n_successes ) )
            choices[i] = choice
        
        return choices
        
        
class Bandits(object):
    
    def __init__(self, probabilities ):
        self.probabilities = probabilities
        
        
    def pull( self, choice):
        return 1 if np.random.random() < self.probabilities[choice] else 0
        
        
        
    def __len__(self):
        return len( self.probabilities )
