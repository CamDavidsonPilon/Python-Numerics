from __future__ import division
#imports and definitions
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet as MCD



class Outlier_detection(object):

    def __init__(self, support_fraction = 0.95, verbose = True, chi2_percentile = 0.995):
        self.verbose = verbose
        self.support_fraction = support_fraction
        self.chi2 = stats.chi2
        self.mcd = MCD(store_precision = True, support_fraction = support_fraction)
        self.chi2_percentile = chi2_percentile
        
    def fit(self, X):
        """Prints some summary stats (if verbose is one) and returns the indices of what it consider to be extreme"""
        self.mcd.fit(X)
        mahalanobis = lambda p: distance.mahalanobis(p, self.mcd.location_, self.mcd.precision_  )
        d = np.array(map(mahalanobis, X)) #Mahalanobis distance values
        self.d2 = d ** 2 #MD squared
        n, self.degrees_of_freedom_ = X.shape
        self.iextreme_values = (self.d2 > self.chi2.ppf(0.995, self.degrees_of_freedom_) )
        if self.verbose:
            print "%.3f proportion of outliers at %.3f%% chi2 percentile, "%(self.iextreme_values.sum()/float(n), self.chi2_percentile)
            print "with support fraction %.2f."%self.support_fraction
        return self

    def plot(self,log=False, sort = False ):
        """
        Cause plotting is always fun.
        
        log: transform the distance-sq to a log ( distance-sq )
        sort: sort the data according to distnace before plotting
        ifollow: a set if indices to mark with yellow, useful for seeing where data lies across views.
        
        """
        n = self.d2.shape[0]
        fig = plt.figure()
        
        x = np.arange( n )
        ax = fig.add_subplot(111)
 
 
        transform = (lambda x: x ) if not log else (lambda x: np.log(x))
        chi_line = self.chi2.ppf(self.chi2_percentile, self.degrees_of_freedom_)     
        
        chi_line = transform( chi_line )
        d2 = transform( self.d2 )
        if sort:
            isort = np.argsort( d2 )    
            ax.scatter(x, d2[isort], alpha = 0.7, facecolors='none' )
            plt.plot( x, transform(self.chi2.ppf( np.linspace(0,1,n),self.degrees_of_freedom_ )), c="r", label="distribution assuming normal" )
            
        
        else:
            ax.scatter(x, d2 )
            extreme_values = d2[ self.iextreme_values ]
            ax.scatter( x[self.iextreme_values], extreme_values, color="r" )
            
        ax.hlines( chi_line, 0, n, 
                        label ="%.1f%% $\chi^2$ quantile"%(100*self.chi2_percentile), linestyles = "dotted" )

        ax.legend()
        ax.set_ylabel("distance squared")
        ax.set_xlabel("observation")
        ax.set_xlim(0, self.d2.shape[0])


        plt.show()

        
