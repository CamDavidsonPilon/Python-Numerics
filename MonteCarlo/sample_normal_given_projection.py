"""
This function generates samples N from N( mu, Simga ) such that N'*nu = x ie. samples 
 N | N*nu = x (which is still normal btw).
 
 Note that actually mu is useless.

"""
import numpy as np
def sample_normal_given_projection(  covariance, x, lin_proj, n_samples=1):
    """
    parameters: 
        x: the value s.t. lin_proj*N = x; scalar
        lin_proj: the vector to project the sample unto (n,)
        covariance: the covariance matrix of the unconditional samples (nxn)
        n_samples: the number of samples to return
        
    returns:
        ( n x n_samples ) numpy array 
        
    """
    variance = np.dot( np.dot( lin_proj.T, covariance), lin_proj )
    
    #normalize our variables s.t. lin_proj*N is N(0,1)

    sigma_lin = np.dot(covariance, lin_proj[:,None])
    cond_mu = ( sigma_lin.T*x/variance ).flatten()
    cond_covar = covariance - np.dot( sigma_lin, sigma_lin.T )/ variance
    
    _samples = np.random.multivariate_normal( cond_mu, cond_covar, size = (n_samples) )
    return ( _samples )
    


        
        
    
    
    