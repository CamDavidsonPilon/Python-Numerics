import numpy as np
import sklearn.linear_model.LinearRegression as LR


class WeightedLinearRegression(LR):
    """
    Implements a weighted least squares class.
    weights: a nx1 vector of non-zero weights. 
    
    """
    def __init__(weights, **kwargs):
        print "warning: untested"
        super(LR, self).__init__(**kwargs)
        self.weights=  weights
        
        
    def fit( X, Y):
        assert X.shape[0] == Y.shape[0] == self.weights.shape[0], "Objects must be same size"
        sqw = np.sqrt( self.weights )
        self.fit( X*sqw, Y*sqw )
        return self
    
    def predict( X ):
        return self.predict( X*np.sqrt(self.weights) )