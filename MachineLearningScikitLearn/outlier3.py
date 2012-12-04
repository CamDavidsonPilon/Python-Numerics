#imports and definitions
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
from numpy import log
import sklearn
from sklearn.datasets import load_diabetes as ld
from sklearn.covariance import MinCovDet as MCD

chi2 = stats.chi2

fdata = ld()
data = fdata.data

#Train the MCD
mcd = MCD(store_precision = True, support_fraction = 0.95)
mcd.fit(data)

#robust estimates
mahalanobis = lambda p: distance.mahalanobis(p, mcd.location_, mcd.precision_  )
d = np.array(map(mahalanobis, data)) #Mahalanobis distance values for the 1000 points
d2 = d ** 2 #MD squared

degrees_of_freedom = data.shape[1]

x = range( len( d2 ))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, d2 )
ax.hlines( chi2.ppf(0.995, degrees_of_freedom), 0, len(d2), label ="99.5% $\chi^2$ quantile", linestyles = "dotted" )

ax.legend()
ax.set_ylabel("distance")
ax.set_xlabel("observation")
ax.set_title( 'Robust detection of outliers at  99.5% $\chi^2$ quantile,\n using MinCovDet algorithm with support fraction of 95%' )

iextreme_values = np.nonzero( d2 > chi2.ppf(0.995, degrees_of_freedom) )
extreme_values = d2[ iextreme_values ]

ax.scatter( iextreme_values, extreme_values, color="r" )

plt.show()


#fuck, put it on a log(log()) scale


fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, log(d2)) 
ax.hlines( log(chi2.ppf(0.995, degrees_of_freedom)), 0, len(d2), label ="99.5% $\chi^2$ quantile", linestyles = "dotted" )

ax.legend()
ax.set_ylabel("log( distance )")
ax.set_xlabel("observation")
ax.set_title( 'Robust detection of outliers at  99.5% $\chi^2$ quantile,\n using MinCovDet algorithm with support fraction of 95%' )

iextreme_values = np.nonzero( d2 > chi2.ppf(0.995, degrees_of_freedom) )
extreme_values = d2[ iextreme_values ]

ax.scatter( iextreme_values, log(extreme_values), color="r" )

plt.show()