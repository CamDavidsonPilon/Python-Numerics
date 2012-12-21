"""
Use MCMC to sample from some copulas


Given a copula, we need to find its pdf. I chose, to establish arbitrary dimensional copulas, to do 
this numerically. I needed to compute the copula differentiated with respect to all of its arguemnts. This
was quite the algorithmic challenge, but I reduced it to a recursive problem that works blazingly fast. This 
felxibility allows us to never have to explicitly find the pdf, which can be difficult even for dimension > 2. 
The differentiation algorithm uses a central difference scheme. Unfortunatly the scheme is unstable for dimensions
greater than 6.

"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy as sp

from mcmc import *
from copulas import *


    
    
mcmc1 = MCMC( lambda u: cdf2pdf( arch_copula, u) , dim = 2, x_0 = np.array( [0.5, 0.5] ) )
mcmc3 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"theta":3}) , dim = 2, x_0 = np.array( [0.5, 0.5] ) )

N = 1000
sampleTheta1 = mcmc1.rvs( N )
sampleTheta3 = mcmc3.rvs( N )

plt.figure()

plt.subplot(221)
plt.scatter( sampleTheta1[:,0], sampleTheta1[:,1], alpha = 0.5)
plt.title("1000 values from a Gumbel \n copula with  %s=1"%r"$\theta$")

plt.subplot(222)
plt.scatter( sampleTheta3[:,0], sampleTheta3[:,1], alpha = 0.5 )
plt.title("1000 values from a Gumbel \n copula with  %s=3"%r"$\theta$")



#lets make the exponential
def make_exp( u ):
    return -np.log(u/3)*3

plt.subplot(223)
plt.scatter( make_exp( sampleTheta1[:,0]) , make_exp( sampleTheta1[:,1] ), alpha = 0.5 )
plt.title("1000 EXP(3) values from a Gumbel \n copula with  %s=1"%r"$\theta$")


plt.subplot(224)
plt.scatter( make_exp( sampleTheta3[:,0]) , make_exp( sampleTheta3[:,1] ), alpha = 0.5 )
plt.title("1000 EXP(3) values from a Gumbel \n copula with  %s=3"%r"$\theta$")

plt.show()


mcmc1 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"f":clayton, "f_inv":inv_clayton}  ) , dim = 2, x_0 = np.array( [0.5, 0.5] ) )
mcmc3 = MCMC( lambda u: cdf2pdf( arch_copula, u, kwargs={"theta":5, "f":clayton, "f_inv":inv_clayton}) , dim = 2, x_0 = np.array( [0.5, 0.5] ) )


dataTheta1 = mcmc1.rvs( N )

dataTheta3 =  mcmc3.rvs( N )

plt.figure()

plt.subplot(221)
plt.scatter( dataTheta1[:,0], dataTheta1[:,1], alpha = 0.5 )
plt.title("1000 values from a Clayton \n copula with %s=1"%r"$\theta$")

plt.subplot(222)
plt.scatter( dataTheta3[:,0], dataTheta3[:,1], alpha = 0.5 )
plt.title("1000 values from a Clayton \n copula with %s=5"%r"$\theta$")



#lets make the exponential
def make_exp( u ):
    return -np.log(u)

plt.subplot(223)
plt.scatter( make_exp( dataTheta1[:,0]) , make_exp( dataTheta1[:,1] ), alpha = 0.5 )
plt.title("1000 EXP(1) values from a Clayton\n copula with %s=1"%r"$\theta$")


plt.subplot(224)
plt.scatter( make_exp( dataTheta3[:,0]) , make_exp( dataTheta3[:,1] ), alpha = 0.5 )
plt.title("1000 EXP(1) values from a Clayton\n copula with %s=5"%r"$\theta$")

plt.show()