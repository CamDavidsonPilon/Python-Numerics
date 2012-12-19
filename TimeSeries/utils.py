#time series utils



def MASE(training_series, testing_series, prediction_series, absolute = "absolute")
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    
    """
    print "Needs to be tested."
    n = training_series.shape[0]
    d = np.abs( np.diff( training_series ) )/(n-1)
    
    if absolute == "absolute":
        errors = np.abs(testing_series - prediction_series )
        return np.mean( errors )/d
    elif absolute == "squares":
        errors = (testing_series - prediction_series)**2
        return np.sqrt( np.mean( errors ) )/d