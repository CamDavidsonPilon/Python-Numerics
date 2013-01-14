import pymc as mc
import numpy as np

#data
X = 5 
Y = 10

#rate = mc.Exponential("rate", 1 ) #priors on N
N = mc.Poisson( "N", 20, value = max(X,Y) )
#N = mc.Uninformative("N", value = max(X,Y) )


pX = mc.Beta("pX", 1,1) #uniform priors 
pY = mc.Beta("pY", 1,1 )


observed = mc.Binomial("obs", p = np.array( [pX, pY] ), n = N, value = np.array( [X,Y] ), observed = True )


