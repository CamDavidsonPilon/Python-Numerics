import inspect

import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from itertools import product


import features

"""
The APRIORI Algorithm for datamining time series.

This algorithm, for lack of better data, will rely on the the pandas library access to data. 
Data will be dataframes like:

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 251 entries, 2011-11-14 00:00:00 to 2012-11-13 00:00:00
    Data columns:
    Open         251  non-null values
    High         251  non-null values
    Low          251  non-null values
    Close        251  non-null values
    Volume       251  non-null values
    Adj Close    251  non-null values
    dtypes: float64(5), int64(1)



S&P Features are generated like this:

    def momentum( series, dataframe ):
        "The momentum for the day is greater than sigma = 1"
        sigma = 1
        if (series["Open"]/series["High"] )/ (series["Low"]/series["Close"] ) > sigma:
            return True

        
A True value is only returned if the feature is present.  In general, the features should be of this form:

    def feature_name( series, dataframe ):
        "Doc String describing what it is, very important so we have human-readable rules later"
        <code>
        return True
        
The arugment series is a pandas series representing the current day.
The argument dataframe is a pandas dataframe representing the prior n days (used for historical features )
        
        
Initialize an instance of Apriori this way:

>> a = Apriori(kwargs)
>> a.add_features( feature1, feature2, feature3,... )
>> a.map(dataframe) #this takes our feature and maps each one to each day.
>> a.find_bivarite_pairs( support = s, confidence = c)
>> a.confident_bivariant_pairs
>> a.predict( a.features_series, a.confident_bivariant_pairs )




>> a.apriori(support = 0.1, confidence = 0.1) #this actually generates the rules
>> 
>> a.rules_ = [ rule1, rule2, rule3...]
where rule1 etc are instances of a Rule class with attributes support, confidence, and important method __str__


"""

class Apriori( object ):
    """
    Implements an association rule mapping system.
    Initialize:
        maxOffset: the maximum number of time periods each feature needs. For example, if considering 
            whether the day exceeds the 20 day moving average, the feature needs information 
            on the past 20 days. Note: This will truncate the resulting features_series to exclude
            maxOffset number of time periods. Default: 25
    
    methods:
        add_features: add a single or list of features. See above to a feature is.
        map: accepts a pandas dataframe and maps the features to each day
        apriori: find the rules in the new data setdefault
        
    attributes:
        rules_: a list of Rules generated from apriori()
        features_series: the product of map() applied to the data frame. Should be a pandas dataframe 
        
    
    >> a = Apriori(kwargs)
    >> a.add_features( [ feature1, feature2, feature2] )
    >> a.map(dataframe) #this takes our feature and maps each one to each day.
    >> a.apriori() #this actually generates the rules
    >>
    >> a.rules_ = [ rule1, rule2, rule3...]
    where rule1 etc are instances of a Rule class with attributes support, confidence, and important method __str__
    """
    
    def __init__(self, verbose = False, maxOffset = 25 ):
        self.verbose = verbose
        
        self.features = []
        self.feature_count = 0
        self.maxOffset = maxOffset
        
    def add_features(self, *args):
        for feature in args:
            feature_object = Feature( feature, id = self.feature_count)
            self.features.append( feature_object)
            self.feature_count +=1 
        if self.verbose:
            print "%d features added."%(len(args) )
        return
        
        
    def map(self, dataframe ):
        
        maxOffset_truncated_index = dataframe.index[ self.maxOffset: ]
        
        features_series = pd.Series( index = maxOffset_truncated_index, dtype=set )
        [ feature.clear() for feature in self.features ] #clear any previous maps we may have done.
        index = dataframe.index
        
        #for date, data in dataframe.iterrows():
        for i,date in enumerate(maxOffset_truncated_index, start = self.maxOffset ):
            if self.verbose and i%75==0:
                print date
        
            current_day_data = dataframe.xs(date)
            previous_data = dataframe.ix[i-self.maxOffset:i]
            features_series[date] = set( [ feature.id for feature in self.features \
                                                if feature(current_day_data, previous_data ) ] )
        
        self.features_series = features_series
        self.instances = self.features_series.shape[0]
        if self.verbose:
            print "Mapping of %d features to %d time periods complete."%(len(self.features), self.instances )
        return
    
    def get_univariate_support(self, n=None):
        """
        Returns the support for each feature as a list. 
        If n is specified, gets the support for the nth feature.
        """
        if n==None:
            return [ float( features.univariate_support)/self.instances for features in self.features ] 
        else:
            return float( self.features[n].univariate_support )/self.instances
    
    def find_bivariate_pairs(self, support = 0.1, confidence = 0.1):
        """
        Finds the most simple rules that satisfy the support/confidence constraint:
           If A_{t-1}, then B_{t} with support S and confidence C
           
        Returns a Rules instance
        
        """
        univariate_support = np.array( self.get_univariate_support() )
        #if a feature does not have sufficitent support, a pair including that feature 
        #will not either, so we need not consider it.
        proper_support = set( np.arange( len(self.features) )[ univariate_support > support ] )
        
        #create a dataframe for current date and tomorrows date.
        paired_dataframe = self._create_paired_dates_dataframe()
        
        pair_counts = {}
        #for pair in product( features_with_proper_support, repeat = 2):
        #    pair_counts[ pair ] = 0 
        
        #this can be MapReduced eventually
        #i = 0
        
        for index, series in paired_dataframe.iterrows():
            #if self.verbose and i%50==0:
            #    print index
            for pair in product( series["current_day"].intersection(proper_support), series["next_day"].intersection(proper_support) ):
                try:
                    pair_counts[ pair ] += 1
                except:
                    pair_counts[ pair ] = 1
            #i+=1
        
        if self.verbose:
            print
            print "Completed calculating bivariate support. %d unique pairs found out of %d total pairs."\
                        %(len(pair_counts), sum (pair_counts.values() ) )
        
        #perform support analysis. Consider the pair only if support > s.
        
        N = self.instances - 1
        support_pair_counts = {}
        for pair, count in pair_counts.iteritems():
            pair_support = float(count) / N #support
            if pair_support > support:
                support_pair_counts[ pair ] = (count, pair_support)
                
        if self.verbose:
            print "Completed support-pruning at %.2f support: %d pairs retained."%( support, len(support_pair_counts) )
              
        rules = Rules()
        for pair, count in support_pair_counts.iteritems():
            raw_count = count[0]
            pair_confidence = float(raw_count)/( N * univariate_support[ pair[0] ] )
            if pair_confidence > confidence:
                rules.append( Rule( priors = (self.features[ pair[0] ],), 
                                                      posteriors = (self.features[ pair[1] ],), 
                                                      confidence = pair_confidence,
                                                      support = count[1], 
                                                    ) )
                
        if self.verbose:
            print "Completed confidence-pruning at %.2f confidence: %d pairs retained."%( confidence, len(rules))
    
        self.confident_bivariate_pairs = rules
        return rules
    
    
    def _create_paired_dates_dataframe( self ):
        d = {"current_day": self.features_series, "next_day": self.features_series.shift(-1) }
        return pd.DataFrame( d).dropna(axis=0)

    
        
    
    def apriori(self, support = 0.1, confidence = 0.1 ):
        pass
    
    
    def reverse_map(self, set ):
        """
        prints the features associated with a set 
        """
        for s in set:
            print self.features[s]
            
    def predict( self, set_of_features, rules_set):
        """
        This returns the predictions for the next time period, given a set of rules.
        
        """
        predictions = Rules()
        for s in set_of_features:
            name = self.features[s].name
            predictions += rules_set.search_for( name, 0 )
            
        return predictions

            
            
class Feature( object ):
    """
    A simple wrapper for features
    """
    def __init__(self, feature_function, id ):
        self.feature_function = feature_function
        self.id = id
        self.rule = feature_function.__doc__
        self.univariate_support = 0
        self.name = feature_function.func_name
        
    def __call__(self,series, dataframe):
        try: 
            v = self.feature_function(series, dataframe)
        except: 
            v = False
        
        if v:
            self.univariate_support += 1
            return True
        else:
            return False
        
    def __str__(self):
        return "Feature class with id %d and rule: \"%s\" "%(self.id, self.rule)

    def clear(self):
        self.univariate_support = 0
        return 
        
        
class Rules(object):
    """
    An aggregator of Rule instances.
    >> rules = Rules()
    >> rules.append(rule1)
    >> rules[0]
    >>   rule1
    >> rules["negvolume"]
    >>   <rules-instance: 1 rule>
    
    
    Search by name by treating like a dictionary.
    
    """
    
    def __init__( self  ):
        self.rules = []
        
    def append(self, x):
        self.rules.append(x)
        
    def __getitem__(self, arg):
        """doubles as a search"""
        try:
            return self.rules[arg]
        except TypeError:
            return self.search_for( arg )
        
    def search_for(self, feature, position = None):
        """This method searches for features in position:
                        0: searches in priors only
                        1: seaches in posteriors only
                        None: searches everywhere
           The feature can be a string (like "momentum") or function. Returns a new Rules instance.
        """
        if type(feature)!=str:
            feature = feature.func_name
        #search on name as string
        _priors_rules = Rules()
        _post_rules = Rules()
        for rule in self.rules:
            if feature in [f.name for f in rule.priors]:
                _priors_rules.append( rule )     
                
            if feature in [f.name for f in rule.posteriors]:
                _post_rules.append( rule )
        
        if position == None:
            return (_priors_rules, _post_rules)
        if position == 0:
            return _priors_rules
        else:
            return _post_rules
            
                
    def __repr__(self):
        return "<Rules-instance: %d rules present>"%len(self.rules)
    def __len__(self):
        return len(self.rules)
        
       
    def __add__(self, rules):
        r = Rules()
        r.rules = rules.rules + self.rules 
        return r
"""
Rule class falicitates the human readable form of a Rule

"""        
class Rule( object ):
    
    def __init__( self, priors, posteriors, support, confidence ):
        self.priors = priors
        self.posteriors = posteriors
        self.support = support
        self.confidence = confidence
        
        
    def __repr__(self):
        str_priors =", \n".join( [ p.rule for p in self.priors ] )
        str_posteriors = ", \n".join( [p.rule for p in self.posteriors] )
        str = "Apriori Rule: " + "(support: %.3f, conf: %.3f) \n"%(self.support, self.confidence)
        
        str += "Priors: \n"
        str += str_priors + "\n"
        str += "Posteriors: \n"
        str += str_posteriors + "\n"
        return str
    
    
    

    
    
if __name__== "__main__":
    
    #dumb hack to take all functions from features.py into here.
    """
    features_to_use = [open_close,
            close_open,
            close_less_than_5D_mv_avg,
            close_less_than_20D_mv_avg,
            close_more_than_5D_mv_avg,
            positiveroc,
            negativeroc,
            negativeMomentum,
            positiveMomentum,
            truerange,
            posvolume,
            negvolume,
            stale_day,
            ]
    """
    features_to_use = [ x[1] for x in inspect.getmembers( features, inspect.isfunction ) ]

    data = DataReader( "GOOG", "yahoo", start = "2011" )

    a = Apriori(verbose = True, maxOffset= 25)
    a.add_features( *features_to_use )
    a.map(data) 
    a.find_bivariate_pairs(support=0.15, confidence=0.65)
    print a.predict( a.features_series.ix[-1], a.confident_bivariate_pairs )[:]