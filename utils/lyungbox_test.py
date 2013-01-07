#lyungBoxTest


import numpy.ma as ma
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats


def LyungBoxTest(ts, tested_lag, significance = 0.95 ):
    """
    ts: a time series.
    tested_lag: is the lag being tested, but must be an int.
    """
    tested_lag = int(tested_lag) 
    f_ts = ts
    f_ts = f_ts - f_ts.mean()
    n = f_ts.shape[0]
    Q = 0
    for i in range(1, tested_lag+1 ):
        lagged_f_ts = f_ts.shift(i)
        m_f_ts = ma.masked_array( lagged_f_ts, mask = np.isnan(  lagged_f_ts ) )
        Q += mstats.pearsonr( f_ts, m_f_ts)[0]**2/(n-i)

    Q = Q*n*(n+2)
    t = stats.chi2(tested_lag).ppf( significance )
    if Q < t:
        print "%d   | Not enough evidence to reject Null: Q = %.4f < %.4f"%(tested_lag, Q,t)
        #print "Not enough evidence to reject "+func.__name__ + " " + series_name+" as not %d autocorrelated according to the Lyung Box test. Q = %.4f < %.4f"%(tested_lag, Q,t)
    else:
        print "%d   | Reject Null: Q = %.4f > %.4f"%(tested_lag, Q,t)
        #print "Reject that "+series_name+ " is autocorrelated at lag %d according to the Lyung Box test test; Q = %.4f > %.4f"%(tested_lag, Q,t)


        
        