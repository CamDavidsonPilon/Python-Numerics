import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression as LR

"""

Note: This shows less than benchmark (all identity) performace. The issue is I am maximizing the wrong thing. I 
should be trying to maximize the partial-correlation. TODO

To do this, we will use a greedy algorithm.

"""

np.seterr( all="raise")

EPSILON = 1e-2

dict_of_transforms = dict([ 
    ("identity", lambda x: x),
    ("logPlus1",lambda x: np.log(x+1)),
    ("sqrtPlus1", lambda x: np.sqrt(x+1) ),
    ("sqrt",lambda x: np.sqrt(x) ),
    ("cuberoot", lambda x: x**(1.0/3.0) ),
    ("squared", lambda x: x**2 ),
    ("squaredPlus1", lambda x: (x+1)**2 ),
    ("cubed",lambda x: x**3 ),
    ("inverse",lambda x: 1./(x+EPSILON) ),
    ("exp",lambda x: np.exp(x) ),
    ("negexp",lambda x: np.exp(-x) ),
    ("inversePlus1", lambda x: 1./(x+1) ),
    ("arctan", lambda x: np.arctan(x) ),
    ("tan", lambda x: np.tan(x) ),
    ("arcsinsqrt", lambda x: np.arcsin(np.sqrt(x)) ),
    ("inversesqrt", lambda x: 1.0/(np.sqrt(x)+EPSILON) ),
    ("inversesqrtPlus1", lambda x: 1.0/(np.sqrt(x+1)) ),
    ("x/(1-x)", lambda x: x/(1-x+EPSILON) ),
    ("sqrtlog",lambda x: np.sqrt( -np.log( x + EPSILON) ) ),
    ("rank", lambda x: np.argsort( x ) ),
    
])    
    


class MaxCorrelationTransformer(object):
    """
    transforms the features of a data matrix to increase the correlation with a response vector, y.
    attributes:
        transforms: a dictionary of functions to try as a transform (defaults to dict_of_transforms)
        normalize01: True is the data will be normalized to between [0,1]
        additional_transforms: a dictionary of transforms in addition to the default.
        
        
    methods:
        fit:
        transform:
        fit_transform:
    
    
    
    """
    def __init__(self,  transforms = dict_of_transforms, 
                        normalize01 = False, 
                        additional_transforms = {}, 
                        verbose=False, 
                        remove_outliers=False
                        tol = 1e-2):
        self.transforms = transforms        
        self.verbose = verbose
        self.transforms.update( additional_transforms )
        #map( _wrapper, transforms + additional_transforms )
        for fname, func in self.transforms.iteritems():
            self.transforms[ fname ] = _wrapper(func, verbose)
        

        self.tol = tol
        
        
    def fit(self, X, Y):
        "to do"
    
        n,d = X.shape
           
        self.transforms_ = ["identity"]*d
        abs_partial_correlations_ = abs(partial_correlation_via_inverse(X,Y)[-1, 0:-1])
        temp_abs_partial_correlations_ = -1e2*np.ones_like( abs_partial_correlations_ )
        ix = np.arange( d) 
        while abs( temp_abs_partial_correlations_.sum() - abs_partial_correlations_.sum() ) > self.tol:
            for i in xrange(d):
                _X = X[:,i]
                Z = X[:, ix != i]
                for transform_name, transform in self.transforms:
                    no_error, f_X = transform( _X )
                    if no_error:
                        pc = abs( partial_correlation( f_X, Y, Z ) )
                        if  pc > abs_partial_correlations_[i]:
                            temp_abs_partial_correlations_[i] = pc
                            self.transforms_[i] = transform_name
            
                            

        return self
    

    def transform(self, X):
        if self.normalize01:
            X = _normalize01( X )
        
        
        newX = X.copy()
        n,d = X.shape
        for i in range(d):
            newX[:,i] = self.transforms[ self.transforms_[i] ]( X[:,i] )
        
        return newX
      
    def fit_transform( self, X, y):
        
        self.fit( X, y)
        return self.transformedX
        

      
def _corr(x,y, remove_outliers=False):
        #check if x,y are same shape
        n = x.shape[0]
        if x.var()==0 or y.var()==0:
            return 0
        else:
            if remove_outliers:
                ee = EllipticEnvelope(store_precision = False, contamination=0.05)
                ee.fit( np.concatenate( [x[:,None],y[:,None] ], axis=1) )
                c = ee.covariance_
                return c[0,1]/np.sqrt( c[0,0]*c[1,1] )
            return np.dot( x - x.mean(), y - y.mean() ) / np.sqrt(( x.var()*y.var() ))/ n
        
def _wrapper(f, verbose = False):
    def g(x):
        try:
            u = f(x)
        except FloatingPointError as e:
            if verbose:
                print "Error.", e
            return False, np.zeros_like(x)
        if ( ~ np.isfinite( u  ) ).sum() > 0:
            if verbose:
                print "Infinite."
            return False, np.zeros_like(x)
        else:
            return True, u
    return g
    
def partial_correlation(X, Y, Z):
    """
    This computes the partial-correlation between X and Y, with covariates Z.
    """
    lr1 = LR()
    lr2 = LR()
    lr1.fit(Z,X)
    lr2.fit(Z,Y)
    
    return np.corrcoef( Y - lr1.predict(Z), X - lr2.predict(Z) )[0,1]
    
def partial_correlation_via_inverse(X, Y=None):
    try:
        X = np.concatenate([ X,Y], axis=1 )
    except:
        pass
    return -cov2corr( np.linalg.inv(np.dot(X.T, X) ) )

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A