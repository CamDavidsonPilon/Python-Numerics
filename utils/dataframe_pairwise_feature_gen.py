
from itertools import combinations_with_replacement, combinations

def create_pairwise_data( df, ignore = [], squares = True):
    """
    df: a dataframe 
    ignore: an iterable of columns to not make quad features out of.
    
    returns:
        a copied dataframe with quadratic features, including squares of variables if squares == True.
    
    """
    n,d = df.shape
    columns = df.columns.diff( ignore )
        
    df = df.copy()
    
    iterator = combinations_with_replacement if squares else combinations  
    
    for x,y in iterator( columns, 2):
        df[ x + "__times__" + y ] = df[x]*df[y]
        
        
    return df
        