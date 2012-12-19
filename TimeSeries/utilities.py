#utilties
import pandas
from pandas.stats.moments import ewma, ewmvar
import numpy as np

def daily_true_range( today, yesterday ):
    return max( today["High"] - today["Low"], abs( today["High"] - yesterday["Close"]), abs( today["Low"] - yeserday["Close"] ) )
    

def average_true_range( dataframe ):
    prev_data = dataframe["Close"].shift(1)
    high = dataframe["High"]
    low = dataframe["Low"]
    avt_range = np.array( map( max, high - low, 
                        abs( high - prev_data), 
                        abs( low - prev_data ) ) )
    return ema( pandas.DataFrame( avt_range, index = dataframe.index ) )
    
    
def ema( series ):
    return ewma( series, span = 20)
    
    
def emv(series):
    return ewmvar(series, com = 9.5)
        
def adx(dataframe):
    upmove = dataframe["High"] - dataframe["High"].shift(1)
    downmove = dataframe["Low"].shift(1) - dataframe["Low"] 
    DM_plus = upmove
    DM_plus[ np.logical_or( downmove > upmove, upmove<0 ) ] = 0
    DM_minus = downmove
    DM_minus[ np.logical_or( upmove > downmove, 0 > downmove) ] =0
    
    DI_plus = 100*ema( DM_plus)/average_true_range( dataframe )
    DI_minus = 100*ema( DM_minus)/average_true_range( dataframe )
    adx = 100*ema( np.abs( DI_plus - DI_minus )/ (DI_plus + DI_minus) )
    return adx
    
    
    
def absolute_returns( series ):
    return (series - series.shift(1))/series.shift(1)
    
 