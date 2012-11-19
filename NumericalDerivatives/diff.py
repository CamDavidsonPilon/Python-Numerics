#numerical high-dim derivatives
import numpy as np
from decimal import Decimal
import decimal


class memorize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}
        
    def __call__(self, *args):
        u = args[1]
        print u
        ustr = u.tostring()
        try:
            return self.cache[ustr]
        except:
            self.cache[ustr] = self.func(*args)
            return self.cache[ustr]
       
   
    def __repr__(self):
        return self.func.__doc__

def _pdf(f, u, delta = 0.001 ):
    n = u.shape[0]
    if n==1:
        t= f(u[0]+delta/2) - f(u[0]-delta/2)
        return t
    else:
        f_plus = lambda *x: f( u[0] + delta/2, *x)
        f_minus = lambda *x: f( u[0] - delta/2, *x)
        return _pdf(f_plus, u[1:], delta ) - _pdf(f_minus, u[1:], delta ) 
        
        
def _pdfOrder4(f, u, delta = 0.001 ):
    n = u.shape[0]
    if n==1:
        t= ( f(u[0]+delta/2) )- ( f(u[0]-delta/2) )
        return t
    else:
        f_plus1 = lambda *x: f( u[0] + delta/2, *x)
        f_plus2 = lambda *x: f( u[0] + delta, *x)
        f_minus1 = lambda *x: f( u[0] - delta/2, *x)
        f_minus2 = lambda *x: f( u[0] - delta, *x)
        p = -_pdfOrder4(f_plus2, u[1:], delta ) + 8*_pdfOrder4(f_plus1, u[1:], delta) \
                    - 8*n(f_minus1, u[1:], delta ) + _pdfOrder4(f_minus2, u[1:], delta )/6
        return p

def cdf2pdf( f, u, delta=0.001, kwargs={} ):
    """numerically unstable for large dimensions"""
    def _wrapper(*args):
        u = np.array(args)
        return f(u, **kwargs)
    n = u.shape[0]
    p= _pdf( _wrapper, u, delta)
    return np.exp( np.log(p) - n*np.log( delta ) )
    #return p / delta**n
    
    