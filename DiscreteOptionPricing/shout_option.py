'''
Created on 2012-01-17

@author: Cameron Davidson-Pilon


'''
import math

def binomial_shoutout(r, u, q, n, x_0, k):
    """
    r: risk-free rate
    u: the return of an up jump
    q: the Risk Neutral probability of an 'up' jump 
    n: the number of periods
    x_0: the start price
    k: the strike price
    
    Explanation of model:
        
        The central idea of my model is based on the recursive formula for a binomial option pricing:
        
                V(t,x) = D*( q*V(t+1, u*x) + (1-q)*V(t+1, x/u) )      (1)
                
        The value of the option at the next time step, V_{t+1}, is dependent on whether the investor chooses to shout or not
        to shout at the current node. The investor tries to maximize his/her profit, thus, I modified the formula above to:
        
                V_t = D*( max ( shout now, shout later) )      (2)
                
        The task is now the calculate the two quantities, 'shout now' and 'shout later'. The value if the investor 
        shouts now is given by (1) with the new payoff max( K-S_T, 0, K-S_t*), and we can calculate the 'shout now' value.
        The 'shout later' requires us to look at the the next value of the nodes in the tree, and calculate the value of these
        nodes given we have NOT shouted.
        
        We can formulize this as:
            
            V(t,x) = D*( max( q*V(t+1, u*x | shout position  = x) + (1-q)*V(t+1, x/u | shout position = x) ,   (3) 
                              q*V(t+1, u*x | haven't shouted ) + (1-q)*V(t+1, x/u | haven't shouted ) ) )   
        
        So when should one shout? Heuristically, one should shout when the expected value of shouting now is greater 
        then the expected value of waiting to shout later, i.e. when 
        
             E[ V(t+1 | shout position = x ]  >  E[ V(t+1 | haven't shouted yet) ]  (4)
        
       Both expectations are under the risk-neutral measure. I tried this heuristic and found that it is optimal 
       to immediately shout if the option is in the money. This makes sense, as if the stock drops, you gain the large 
       K-S_T payoff, but if the stock rises you are protected and still receive, albeit small, K-S_t* > 0. Obviously, 
       if the stock is not in the money it is pointless to shout.
    
        
    """

    R = float(r); U = float(u); Q = float(q); X_0 = float(x_0); K = float(k); N = float(n)
    D = math.exp(-R/N)
    dictionary={}
    shout_times = []
    
    def payoff(K,x,m):
        #This is a put-style payoff
        return max(K-x, K-m, 0)
    
    
    def value(n, x, m):
        """ find the value of a shout put"""
        try:
           return dictionary["%s,%s,%s"%(n,x,m)]
        except:
            if n==N:
                return payoff(K,x,m)
            else:
                shout_now = Q*value(n+1, U*x, m) + (1-Q)*value(n+1,x/U, m)
                if m==x:
                    shout_later = Q*value(n+1, U*x, U*x) + (1-Q)*value(n+1, x/U, x/U) 
                else:
                    shout_later = 0
                    
                if shout_now>shout_later:
                    #This is the condition when to shout. If true, add it shout_times
                    if f(m,n) not in shout_times:
                        shout_times.append( f(m,n) )
                        
                y = D*max( shout_now, shout_later )
                dictionary[ "%s,%s,%s"%(n,x,m) ] = y
                
                return y
                
    def f(x,n):
        """ This is to find the number of up jumps given a price x and time period n."""
        return (n,int(0.5*(math.log(x/X_0,U)+n)))
    
    def delta(n,k):
        """This function computes the delta at each node (n,k), where n is the number 
        of up jumps and n is the time period"""
        up = X_0*u**(2*k-n+1)
        down = X_0*u**(2*k-n-1)
        return ( value(n,up,up)-value(n,down,down) )/(up-down)

    print value(0,X_0,X_0)
    for s in shout_times:
        print s

"""
Example:

Assume the following:
(i) S(0) = K = 1, and the volatility of the underlying security is  sigma = 40%.
(ii) The continuously compounded interest rate is constant and equal to r = 1%.
(iii) The maturity of the contract is 12 months, and the owner can "shout" only at the end of each
month.
(iv) The underlying security pays no dividends.
Using a binomial tree model with 12 time periods, find the value of this option at time zero. Identify
all nodes at which it is optimal for the owner to "shout" and find the replicating portfolio at time 0.

From this, we need the size of an "up" stock movement, and the risk-neutral probability of an "up" movement.
r = 0.01
sigma = .40
periods = 12 
S_0 = K = 1


u = exp(sigma/sqrt(periods) )
q = (exp(r/periods) - 1/u)/(u - 1/u)

binomial_shoutout(r, u, q, periods, S_0, K)


   