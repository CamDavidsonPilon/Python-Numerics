#features

import utilities


def open_close( current_day_data, previous_data ):
    """Open > close"""
    if current_day_data["Open"]>current_day_data["Close"]:
        return True
        
def close_open( current_day_data, previous_data ):
    """Close > open"""
    if current_day_data["Open"]<current_day_data["Close"]:
        return True
        
        
def close_more_than_5D_mv_avg( current_day_data, previous_data):
    "The close greater than the 5 day average."
    if utilities.ema(previous_data["Close"][-5:])[-1] < current_day_data["Close"]:
            return True
            
def close_less_than_5D_mv_avg( current_day_data, previous_data):
    "The close is less than the 5 day average."
    if utilities.ema(previous_data["Close"][-5:])[-1] > current_day_data["Close"]:
            return True

def close_less_than_2D_avg( current_day_data, previous_data):
    "The close is less than the 2 day average."
    if previous_data["Close"][-2:].mean() > current_day_data["Close"]:
            return True
            
def close_greater_than_2D_avg( current_day_data, previous_data):
    "The close is greater than the 2 day average."
    if previous_data["Close"][-2:].mean() < current_day_data["Close"]:
            return True

# def close_less_than_20D_mv_avg( current_day_data, previous_data):
    # "The close is less than the 20 day average."
    # if previous_data["Close"][-20:].mean() > current_day_data["Close"]:
            # return True
            


def positiveMomentum( series, dataframe):
    "The stochastic oscillator over the last 5 days is greater than 1"
    H = dataframe["Close"][-5:].max()
    L = dataframe["Close"][-5:].min()
    if ( series["Close"] - L)/(H-L) >1:
        return True
        
def negativeMomentum( series, dataframe):
    "The stochastic oscillator over the last 5 days is less than 0.5"
    H = dataframe["Close"][-5:].max()
    L = dataframe["Close"][-5:].min()
    if ( series["Close"] - L)/(H-L) < 0.5:
        return True
    

def positiveroc( series, dataframe):
    "The rate of change from 5 days ago is more than 3%"
    if (series["Close"] - dataframe["Close"].ix[-5])/(dataframe["Close"].ix[-5]) > 0.03 :
        return True
        
def negativeroc( series, dataframe):
    "The rate of change from 5 days ago is less than -3%"
    if (series["Close"] - dataframe["Close"].ix[-5])/(dataframe["Close"].ix[-5]) < -0.03 :
        return True
            
def posvolume( series, dataframe ):
    "The volume exceeded it's 5 day average"
    if series["Volume"] > dataframe["Volume"][-5:].mean():
        return True
        
def negvolume( series, dataframe ):
    "The volume fell below it's 5 day average"
    if series["Volume"] < dataframe["Volume"][-5:].mean():
        return True
            
def stale_day( series, dataframe):
    "The |open - close|/open < 0.005"
    if abs( series["Open"] - series["Close"] )/series["Open"] < 0.005:
        return True
        

        
def binary_momentum_bull(series, dataframe):
    "The past two days have been positive."
    if (series["Close"] > series["Open"]) and (dataframe.ix[-1]["Close"] > dataframe.ix[-1]["Open"] ):
        return True
        
def binary_momentum_bear(series, dataframe):
    "The past two days have been negative."
    if (series["Close"] < series["Open"]) and (dataframe.ix[-1]["Close"] < dataframe.ix[-1]["Open"] ):
        return True

        
def adx_greater_40( series, dataframe):
    """The ADX is greater than 40 => strong trend."""
    dataframe.append( series )
    if utilities.adx( dataframe ).ix[-1] > 40:
        return True
        
def adx_less_20( series, dataframe):
    """The ADX is less than 20 => weak trend """
    dataframe.append( series )
    if utilities.adx( dataframe ).ix[-1] < 20:
        return True

def volatility_increasing( series,  dataframe ):
    """EMV is increasing """
    dataframe.append( series )
    returns = utilities.absolute_returns( dataframe["Close"] )
    vol = utilities.emv( returns )
    if vol.ix[-1] > vol.ix[-2]:
        return True
def volatility_decreasing( series,  dataframe ):
    """EMV is decreasing """
    dataframe.append( series )
    returns = utilities.absolute_returns( dataframe["Close"] )
    vol = utilities.emv( returns )
    if vol.ix[-1] < vol.ix[-2]:
        return True

def nightly_gains( series, dataframe):
    """The open was greater than yesterdays close"""
    if series["Open"] > dataframe["Close"][-1]:
        return True
def nightly_losses( series, dataframe):
    """The open was less than yesterdays close"""
    if series["Open"] < dataframe["Close"][-1]:
        return True

#def breakout_support( series, dataframe ):
    
    