#Least squares with penalty on wrong sign.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymc as mc
import scipy.optimize as sop

def sign(x):
    return -1 if x<0 else 1

def loss( y, yhat, coef = 100):
    """vectorized"""
    sol = np.zeros_like(y)
    ix = y*yhat < 0 
    sol[ix] = coef*yhat**2 - sign(y[ix])*yhat + abs(y[ix])
    sol[ ~ix ] = abs( y[~ix] - yhat )
    return sol

        
#generate some artifical data
size = 250
beta = 0.4
alpha = 0.0

X = np.random.randn( size )
Y = beta*X + alpha + np.random.randn( size )



# Form the bayesian analysis.
prec = mc.Uniform( "prec", 0, 100 )
beta_0 = mc.Normal( "beta", 0, 0.0001 )
alpha_0 = mc.Normal( "alpha", 0, 0.0001 )


@mc.deterministic
def mean( X = X, alpha_0 = alpha_0, beta_0 = beta_0 ):
    return alpha_0 + beta_0*X
    
to_predict_x = np.linspace( -10, 10, 100)

    
obs = mc.Normal( "obs", mean, prec, value = Y, observed = True)

model = mc.Model( {"obs":obs, "beta_0":beta_0, "alpha_0":alpha_0, "prec":prec} )
mcmc = mc.MCMC( model )

n_samples = 100000
burnin = 50000
mcmc.sample( burnin + n_samples, burnin)
mean_alpha_0 = mcmc.alpha_0.stats()["mean"] #correspondes to the least squares estimate
mean_beta_0 = mcmc.beta_0.stats()["mean"] #correspondes to the least squares estimate
ls_prediction = mean_alpha_0 + mean_beta_0*to_predict_x


alpha_trace = mcmc.alpha_0.trace.gettrace()
beta_trace = mcmc.beta_0.trace.gettrace()
rprec = [1.0/np.sqrt(prec.random()) for i in range(n_samples ) ]
norm_samples = rprec*np.random.randn(n_samples)


v = np.zeros_like( to_predict_x)
for i,x in enumerate(to_predict_x):
    post_samples = norm_samples + (alpha_trace + beta_trace*x)
    tomin = lambda yhat: loss( post_samples, yhat).mean()
    v[i] = sop.fmin( tomin, ls_prediction[i] )
    
print v
    
#nice plots
plt.figure()
plt.plot( to_predict_x, ls_prediction, lw =2, label = "Least squares prediction", c="k" )
plt.plot( to_predict_x, v, lw = 2, label = "Bayesian Loss-optimized prediction", c= "r")
plt.scatter( X, Y, alpha = 0.4 )
plt.legend()
plt.title("Least squares predictions vs \n Bayesian Loss-optimized predictions")
plt.xlim(-7, 7)
plt.ylim(-5, 5)
plt.savefig( "LossOptII.png" )

    

