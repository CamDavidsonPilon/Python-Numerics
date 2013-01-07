#risk measures

import scipy.stats as stats
from scipy.optimize import fsolve
import numpy as np



def VaR(ts, alpha, flavour):
    if flavour == "historical":
        temp_ts = ts.copy()
        temp_ts.sort()
        n = len( temp_ts)
        try:
            return -temp_ts.values[ np.floor( (1-alpha)*n ) ]
        except:
            return -temp_ts[ np.floor( (1-alpha)*n ) ]
            
    elif flavour == "t":
        t = stats.t
        t = stats.t( *t.fit( ts ) )
        return -t.ppf( 1-alpha )
            
    elif flavour == "normal":
        mean = ts.mean()
        std = ts.std()
        return -stats.norm.ppf( 1-alpha, mean, std )
    elif flavour == "Cornish-Fischer":
        z_c = -stats.norm.ppf( 1-alpha, 0 ,1)
        S = stats.skew(ts)
        K = stats.kurtosis(ts)
        z_cf = z_c + (z_c**2-1)*S/6 + (z_c**3- 3*z_c)*K/24 + (2*z_c**3-5*z_c)*S**2/36
        return ts.mean() - z_cf*np.sqrt( ts.std() )
        
    elif flavour == "kernel":
        kde = stats.gaussian_kde( ts )  
        print kde.factor

        f = lambda x: kde.integrate_box_1d(-1, x) - (1-alpha)
        return -fsolve( f, -0.05)[0]
        
        
        
def ES( ts ,alpha, flavour="historical"):
    var = VaR( ts, alpha, flavour)
    n_simulations = 200000
    if flavour=="historical":
        return -ts[( ts < -var )].mean()
        
    elif flavour == "normal":
        mean = ts.mean()
        std = ts.std()
        norm = stats.norm( mean, std )
        samples = -norm.rvs( n_simulations )
        
        return samples[ var <= samples ].mean()
        
    elif flavour == "t":
        t = stats.t
        t = stats.t( *t.fit( ts ) )
        samples = -t.rvs( n_simulations )
        return samples[var <=samples ].mean()
    
    elif flavour == "kernel":
        kde = stats.gaussian_kde(ts)
        samples = -kde.resample(n_simulations)
        return samples[ var<= samples].mean()