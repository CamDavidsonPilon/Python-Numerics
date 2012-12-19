import numpy as np
from sklearn.covariance import EllipticEnvelope

"""

Note: This shows less than benchmark (all identity) performace. The issue is I am maximizing the wrong thing. I 
should be trying to maximize the partial-correlation. TODO

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
    def __init__(self,  transforms = dict_of_transforms, normalize01 = False, additional_transforms = {}, verbose=False, remove_outliers=False ):
        self.transforms = transforms        
        self.verbose = verbose
        self.transforms.update( additional_transforms )
        #map( _wrapper, transforms + additional_transforms )
        for fname, func in self.transforms.iteritems():
            self.transforms[ fname ] = _wrapper(func, verbose)
        
        self.normalize01 = normalize01
        self.remove_outliers = remove_outliers
        
        
    def fit(self, X, Y):

        n,d = X.shape
        self._X = X
        if self.normalize01:
            X = _normalize01(X)
           
        self.transforms_ = ["identity"]*d
        self.correlations_ = [0]*d
        #change this so it is inline with the dictionary above.
        for i in xrange(d):
            x = X[:, i]
            max_cor = abs(_corr(x, Y))
            epsilon = 0.05
            for fname, func in self.transforms.iteritems():
                #cor = np.corrcoef( func(x)[:,None].T, Y.T )[1,0]
                fx = func(x)

                    
                cor = _corr( fx, Y, self.remove_outliers)
                if abs(cor) > max_cor + epsilon :
                    self.correlations_[i] = cor
                    max_cor = abs(cor)
                    self.transforms_[i] = fname
                    epsilon = 0
                    
        self.transformedX = self.transform( X )
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
            return np.zeros_like(x)
        if ( ~ np.isfinite( u  ) ).sum() > 0:
            if verbose:
                print "Infinite."
            return np.zeros_like(x)
        else:
            return u
    return g
    
    
def _normalize01(X):
    
    newX = X.copy()
    
    newX = newX - newX.min(axis=0)
    newX = newX/newX.max(axis=0)
    return newX
    