"""
use this decorator for recursion to cache calls

"""


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
        
        