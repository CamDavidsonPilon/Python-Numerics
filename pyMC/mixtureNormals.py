from pymc import *

size = 20
p = Uniform( "p", 0 , 1)

ber = Bernoulli( "ber", p = p, size = size)

precision = Gamma('precision', alpha=0.1, beta=0.1)

mean1 = Normal( "mean1", 0, 0.001 )
mean2 = Normal( "mean2", 0, 0.001 )

@deterministic
def mean( ber = ber, mean1 = mean1, mean2 = mean2):
    return ber*mean1 + (1-ber)*mean2
   

#generate some artifical data   
v = np.random.randint( 0, 2, size)
data = v*(10+ np.random.randn(size) ) + (1-v)*(-10 + np.random.randn(size ) )


obs = Normal( "obs", mean, precision, value = data, observed = True)

model = Model( {"p":p, "precision": precision, "mean1": mean1, "mean2":mean2, "obs":obs} )