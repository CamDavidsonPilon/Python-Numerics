"""
This implements the Theil-Sen linear regression estimator for 2d data points.
The jist of it is:
It returns the median all computed slope value between pairs (x_i, y_i), (x_j, y_j), (x_i > x_j)
where slope = (y_i - y_j)/(x_i - x_j)


Very robust to outliers.

"""
import numpy as np
import bottleneck #very fast searching and sorting written in Cython.
import itertools

def theil_sen(x,y, sample= "auto", n_samples = 1e7):
    """
    Computes the Theil-Sen estimator for 2d data.
    parameters:
        x: 1-d np array, the control variate
        y: 1-d np.array, the ind variate.
        sample: if n>100, the performance can be worse, so we sample n_samples.
                Set to False to not sample.
        n_samples: how many points to sample.
    
    This complexity is O(n**2), which can be poor for large n. We will perform a sampling
    of data points to get an unbiased, but larger variance estimator. 
    The sampling will be done by picking two points at random, and computing the slope,
    up to n_samples times.
    
    """
    assert x.shape[0] == y.shape[0], "x and y must be the same shape."
    n = x.shape[0]
    
    if n < 100 or not sample:
        ix = np.argsort( x )
        slopes = np.empty( n*(n-1)*0.5 )
        for c, pair in enumerate(itertools.combinations( range(n),2 ) ): #it creates range(n) =( 
            i,j = ix[pair[0]], ix[pair[1]]
            slopes[c] = slope( x[i], x[j], y[i],y[j] )
    else:
        i1 = np.random.randint(0, n, n_samples)
        i2 = np.random.randint(0, n, n_samples)
        slopes = slope( x[i1], x[i2], y[i1], y[i2] )
        #pdb.set_trace()
    
    slope_ = bottleneck.nanmedian( slopes )
    #find the optimal b as the median of y_i - slope*x_i
    intercepts = np.empty( n )
    for c in xrange(n):
        intercepts[c] = y[c] - slope_*x[c]
    intercept_ = bottleneck.median( intercepts )

    return np.array( [slope_, intercept_] )
        
        
        
def slope( x_1, x_2, y_1, y_2):
    return (1 - 2*(x_1>x_2) )*( (y_2 - y_1)/np.abs((x_2-x_1)) )
    
    
    
    
if __name__=="__main__":
    x = np.asarray( [ 0.0000, 0.2987, 0.4648, 0.5762, 0.8386 ] ) 
    y = np.asarray( [ 56751, 57037, 56979, 57074, 57422 ] ) 
    print theil_sen( x, y )
    