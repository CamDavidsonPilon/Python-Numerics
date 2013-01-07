
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def qq_plot( data ):
    plt.figure()
    (osm, osr) = stats.probplot( data, sparams=[ data.mean(), data.std()], dist='norm', fit = True )
    x_ = np.array( [min( osm[1] ), max (osm[1] ) ] )
    slope = osr[0]
    inter = osr[1]
    
    plt.plot( x_, x_ , label="Line y=x")
    plt.scatter( osm[0], osm[1] )
    plt.xlabel( "Observed" )
    plt.ylabel( "Theoretical" )
    plt.show()