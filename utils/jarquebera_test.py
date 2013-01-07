#jaque-berra test

import scipy.stats as stats

def JarqueBeraTest(data,significance = 0.95):
    """
    If the data come from a normal distribution, the JB statistic asymptotically has a chi-squared distribution with two degrees of freedom, 
    so the statistic can be used to test the hypothesis that the data are from a normal distribution. 
    
    """
    n = data.shape[0]
    if n < 2000:
        print "Warning: JarqueBera tests works best with large sample sizes (> ~2000 )."
    
    S = float(n)/6*( stats.skew(data)**2 + 0.25*(stats.kurtosis( data, fisher=True) )**2)
    t = stats.chi2(2).ppf( significance )
    if S < t:
        print "Not enough evidence to reject as non-Normal according to the Jarque-Bera test. S = %.4f < %.4f"%(S,t)
    else:
        print "Reject that is Normal according to the Jarque-Bera test; S = %.4f > %.4f"%(S,t)
        