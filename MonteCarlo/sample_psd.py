

def generate_psd_matrix( dim, avg_covariance=0, diag=np.array([]) ):
    """
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
            variance = diag[i] if diag.any() else np.abs(avg_covariance + np.random.randn(1) ) 
            covariance = avg_covariance + np.random.randn(i)
            
            #check if det > 0 of matrix | A   cov |
            #                           | cov var |
            if i==0 or (variance - np.dot( np.dot( covariance.T, invA), covariance ) ) > 0:
                b_flag = True
        M[i, :i] = covariance
        M[:i, i] = covariance
        M[i,i] = variance
        
        if i > 0:
            invA = invert_block_matrix_CASE1( A , covariance, variance, invA)
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
    
    
        