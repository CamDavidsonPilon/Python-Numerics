import numpy as np
from grammschmidt import gs
import warnings
import scipy.stats as stats

def sample_pd_matrix( dim, avg_variance = 1, diag = np.array([]) ):
    """
    avg_variance = the average variance, scalar. 
    dim: the dimension of the sampled covariance matrix
    diag: enter a dim-dimensional vector to use as the diagonal eigenvalues elements.
    """
    
    #create an orthonormal basis 
    Ob = gs(np.random.randn( dim,dim ) )
    if not diag.any():
        """
        This uses the fact that the sum of varinaces/n == Trace(A)/n == sum of eigenvalues/n ~= E[ Gamma(1, 1/avg_variance) ] = avg_variance
        """
        diag = stats.gamma.rvs( 1, scale = avg_variance, size = ( (dim,1) ) )
    else:
        diag = diag.reshape( (dim,1) )
    return np.dot( Ob.T*diag.T, Ob )
    

def return_lower_elements(A):
    n = A.shape[0]
    t = [ (i,j) for j in range(0,n) for i in range(j+1,n) ]
    return np.array( [A[x] for x in t] )
    
    
def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func
    
@deprecated    
def generate_pd_matrix( dim, avg_covariance=0, avg_variance = 0, diag=np.array([]) ):
    """
    Currently unstable for dim > 25. I would not use.
    
    
    This uses Sylvester's criterion to create n-dim covariance (PSD) matrices.
    To make correlation matrices, specify the diag parameters to be an array of all ones.
    parameters:
        avg_covariance: is added to a Normal(0,1) observation for each covariance.
            So, the sample mean of all covariances should be avg_covariance.
        dim: the dimension of the sampled covariance matrix
        diag: enter a dim-dimensional vector to use as the diagonal elements.

    """
    invA = None
    M = np.zeros( (dim,dim) )
    for i in xrange( 0, dim ):
        A = M[:i,:i]
        b_flag = False
        while not b_flag:
            #generate a variance and covariance array
            variance = diag[i] if diag.any() else avg_variance + np.abs( np.random.randn(1) ) 
            covariance = (avg_covariance + np.random.randn(i))  #for stability
            #pdb.set_trace()
            #Using Danny's algorithm
            if i > 0:
                c = variance*np.random.rand(1) # > 0, < variance 
                _lambda = np.dot( np.dot( covariance[:,None].T, invA), covariance[:,None] )[0] +1 
                print _lambda
                covariance = (np.sqrt(c)/np.sqrt(_lambda))*covariance.T
                
            
            #check if det > 0 of matrix | A   cov |
            #                           | cov var |
            
            
            if i==0 or _lambda > 0:
                b_flag = True
        M[i, :i] = covariance
        M[:i, i] = covariance
        M[i,i] = variance
        
        if i > 0:
            invA = invert_block_matrix_CASE1( A , covariance, variance, invA)
            #invA = np.linalg.inv( M[:i+1,:i+1])
        else:
            invA = 1.0/M[i,i]
        
    return M
  

      
def invert_block_matrix_CASE1( A, b, c, invA = None):
    """
    Inverts the matrix | A  b |
                       | b' c |
    
    where A is (n,n), b is (n,) and c is a scalar
    P,lus if you know A inverse already, add it to make computations easier.
    This is quicker for larger matrices. How large?
    
    """
        
    n = A.shape[0]    
    if n == 1 and A[0] != 0:
        invA = 1.0/A
    if b.shape[0] == 0:
        return 1.0/A
    
    if invA == None:
       invA = np.linalg.inv(A)

    inverse = np.zeros( (n+1, n+1) )
    k = c - np.dot( np.dot( b, invA), b )

    inverse[ :n, :n] = invA  + np.dot(  np.dot( invA, b[:,None]), np.dot( b[:,None].T, invA) )/k
    inverse[n, n] = 1/k
    inverse[:n, n] = -np.dot(invA,b)/k
    inverse[n, :n] = -np.dot(invA,b)/k
    return inverse
    
    
        