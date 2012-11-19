"""
This is a simple, recursive, implementation of pricing
a option with uncertain volatility (known sigma_max, sigma_min) in a 
recombining trinomial tree model.

It is surprisingly fast, thanks to cacheing the calls.

example of use below.

"""

import numpy as np

class memorize( object ):

    def __init__(self, func):
        self.func = func
        self.cache = {}
        
    def __call__(self, *args):
        try:
            return self.cache[args]
        except:
            self.cache[args] = self.func(*args)
            return self.cache[args]

    def __repr__(self):
        return self.func.__doc__

def Snj( S_0, n ,j, sigma_max, r, t_delta):
    return S_0*np.exp( j*sigma_max*np.sqrt(delta_t) + n*r*delta_t )

@memorize    
def price( style, F, sigma_max, sigma_min, delta_t, r, S_0, n, j, N):
    """
    This is the main function.
        style: either "min" or "max", get the min or maximum price rescpectivly.
        F: the final payoff function
        sigma_max, sigma_min: the max and min volatility
        delta_t: the length of time step
        r: the risk-free rate
        S_0: the initial price of the underlying
        n: the time step
        j: position in tree
        N: the number of time steps. I'd keep this not too large, else you stack overflow lol.
    """
    if n == N:
        return F( Snj(S_0, n, j, sigma_max, r, delta_t ) )

    t = sigma_max*np.sqrt(delta_t)/2
    l = (1-t)*price(style,F, sigma_max, sigma_min, delta_t, r, S_0, n+1, j+1, N) + \
        (1+t)*price(style,F, sigma_max, sigma_min, delta_t, r, S_0, n+1, j-1, N) - \
        2*price(style,F, sigma_max, sigma_min, delta_t, r, S_0, n+1, j, N)
    
    c = 0.5 if (1-2*(style=="min"))*l >= 0 else sigma_min**2/(2*sigma_max**2)

    return  np.exp( -r*delta_t)*( price(style, F, sigma_max, sigma_min, delta_t, r, S_0, n+1, j, N) + c*l )
    
    
    
if __name__=="__main__":
    
    def F(x):  
        # a collared option.
        return max(0, x - 100) - max( 0, x - 120)

    sigma_max = 0.4
    sigma_min = 0.1
    r= 0.1
    S0 = 100.
    N = 100.
    delta_t = 1.0/N
    
    print price("min", F, sigma_max, sigma_min, delta_t, r, S0, 0,0, N )
    print price("max", F, sigma_max, sigma_min, delta_t, r, S0, 0,0, N )
    
    """
    4.54345306389
    12.358008422
    """