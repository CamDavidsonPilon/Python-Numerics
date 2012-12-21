#imports and definitions
import numpy as np
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles
from sklearn.covariance import MinCovDet as MCD

chi2 = stats.chi2

data, target = make_circles(200, noise=.1, shuffle=True,factor=.6)

target[ np.nonzero( target==1)[0][0] ] = 0

on = target==1
z = target==0
p = on.sum()/200.0
print p
mcd = MCD(store_precision = True, support_fraction = p)
mcd.fit(data)

#robust estimates
mahalanobis = lambda p: distance.mahalanobis(p, mcd.location_, mcd.precision_  )
d = np.array(map(mahalanobis, data)) #Mahalanobis distance values for the 1000 points
d2 = d ** 2 #MD squared

degrees_of_freedom = data.shape[1]

x = np.arange( len( d2 ))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter( x[on], d2[on], c="r", label="class 1" )
ax.scatter( x[z], d2[z], c="b", label="class 0" )


#ax.hlines( chi2.ppf(0.5, degrees_of_freedom), 0, len(d2), label =str(50)+"% $\chi^2$ quantile", linestyles = "dotted" )

ax.legend()
ax.set_ylabel("distance")
ax.set_xlabel("observation")

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter( data[on,0], data[on,1], c="r", label="class 1" )
ax.scatter( data[z,0], data[z,1], c="b", label="class 0" )

ax.set_title("2-D non-linearly seperable data")


plt.show()






