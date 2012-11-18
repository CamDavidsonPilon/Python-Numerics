import re
import numpy as np

import string

class EncodingScheme(object):
        """
        EncodingScheme is a class to make Markov model data out of raw data. 
        Input:
            list_of_regex_bins: a list of regular expressions, as strings, representing how to "bin"
                the raw data. eg: [ '[0-9]', '[a-z]', '[A-Z]' ]
                A -1 is inserted if the item cannot be binned correctly. 
                Notes: -Try not to overlap bins. 
                       -To specify all unique bins, leave the list empty, [].
                       -An exception is thrown if a item is not able to be binned.
                       #-To specify some bins, and have everything else unique, 
            to_append_to_end:
                if the series data is not the same length ( eg: password data), this specifies what to append
                to end before performing analysis. If not needed, leave as None. This is still buggy.
                
            garbage_bin: a boolean to include a garbage bin, ie a bin that collects everything not collected.
                Notes: having garbage_bin to True is pretty much useless if all unique bins is set, ie. 
                       having list_of_regex_bins = []
            
            restict: only allow series with elements that belong to this iterable. (For example: string.printable or string.ascii_letters)
            
        attributes:
            unique_bins: a dictionary of the bins used to encode the data and the encode mapping.
            realized_bins: a dictionary of the bins used to encode the realized data values that
                                satisfy the bins. Useful for debugging and seeing what garbage is collected
                                with realized_bins['garbage']
         
        Methods:
            encode(raw_data): returns the encoded data as a generator.
            
        """


        def __init__(self, list_of_regex_bins=[], to_append_to_end = None, garbage_bin=False, restrict=None  ):
            self.list_of_regex_bins = list_of_regex_bins
            self.to_append_to_end = to_append_to_end
            self.unique_bins = dict()
            self.number_of_bins = -1
            self.garbage_bin = garbage_bin
            self.realized_bins = dict( zip ( self.list_of_regex_bins, [ set() for i in xrange(len(list_of_regex_bins) )  ] ) )
            self.restrict = restrict
            
            
        def encode(self, data):
            """
            This function creates a representation of the data.
            Input:
                data: a list of iterables (eg: strings, lists, arrays, np.arrays).
            output:
                a generator of 1d numpy arrays of computer-readable time series, starting at 0 to unique_number of elements.
                
            ex:
                eScheme = Encoding_scheme( )
                input = ['data', 'atad', 'dta']
                eScheme.encode( data)
                
            """
            self._init_encode(data)
            
            idata = self.yield_data() #returns a generator
            self._create_dict(idata)
            
            idata = self.append_ends( self.data, self.series_length ) #returns a generator

            return self._encode_generator(idata)

 
        
        def _encode_generator(self, idata):
           for series in idata:
                if self.restrict:
                    for item in series:
                        if item not in self.restrict:
                            try:
                                next(idata)
                                break
                            except:
                                pass

                encoded_data = np.zeros( self.series_length, dtype="int" ) 
                for col_i, item in enumerate(series):
                    encoded_data[col_i] = self._encode( item )
                yield encoded_data
        
        
        def _create_dict(self, idata):
            for series in idata:
                if self.restrict:
                    for item in series:
                        if item not in self.restrict:
                            print series
                            try:
                                next(idata)
                                break
                            except:
                                pass

                        
                for item in series:
                    self._encode( item )
        
        def _init_encode(self, data):
            self.data = data
            self.series_length = self._max_length(data)

        
        def decode(self, sample):
            try:
                return "".join([ self.inv_map[s] for s in sample ])
            except:
                self.inv_map = dict((v,k) for k, v in self.unique_bins.iteritems())
                return "".join([ self.inv_map[s] for s in sample ]) 
        
        
        def _max_length(self,data):
            return max( map( len, data ) )
        
        def yield_data(self):
            for series in self.data:
                
                
                yield series
        
        def append_ends( self, data, length):
            for series in data:
                if len(series)<length:
                        series+= self.to_append_to_end*(length-len(series) ) #this is too specific
                yield series

            
        def _encode(self, item):
            """
            This both creates the dictionares/bins and returns the proper encoding.
            
            
            """
            
            if not self.list_of_regex_bins:
                #more efficient in python to use try-else
                try:
                    return self.unique_bins[str(item)]
                except KeyError:
                    #This won't distinguish 1.0 from 1 etc.
                    self.number_of_bins +=1
                    self.unique_bins[str(item)] = self.number_of_bins 
                    return self.unique_bins[str(item)]
            
            else:
                for regex in self.list_of_regex_bins:
                    if re.match( regex, str(item) ):
                        try:
                            return self.unique_bins[regex]
                        except KeyError:
                            self.number_of_bins +=1
                            self.unique_bins[regex] = self.number_of_bins 
                            self.realized_bins[regex].add( item)
                            return self.unique_bins[regex]
                            
                #we didnt collect it. See if garbage bin is enabled.
                if self.garbage_bin:
                    if 'garbage' not in self.unique_bins.keys():
                        self.number_of_bins +=1
                        self.unique_bins['garbage'] = self.number_of_bins
                        self.realized_bins['garbage'] = set()
                        
                    self.realized_bins['garbage'].add( item)           
                    return self.unique_bins['garbage']
                    
                else:
                    raise BinningError(item)
                
                
                
                
                
                
class BinningError( Exception):
    #thrown if no bin is found for some value.
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return "Could not find a bin for value %s."%repr(self.value)
            
            
            
            
            
                    
            
            
        
            
            
            
        
            
        