
import numpy as np
import encoding

class MultinomialMM(object):
    """
    Create and learn a  multinomial Markov model 
    Input:
        encoding: a EncodingScheme class that will process the data prior to fitting. If
                  no scheme is given, and the data is inputed without encoding, a default 
                  encoding will be used (all unique binning).
    
    Attributes:
        self.data: the data used to fit the model
        self.unique_elements: the found unique elements of the data
        self.init_probs_esimate: the probability vector of inital emissions
        self.trans_probs_estimate: the trasmission probability matrix of going from 
            emission [row] to emission [col].
    
    Methods:
        self.fit(data, encoded=True)
        self.sample( n=1)
        self.decoded_sample(n=1)
        
    
    
    """
    def __init__(self, encoding=None):
        self.encoding = encoding
        
       
    def fit(self, data, encoded=True):
        """
        Fit the model to some data. Estimates the transition and intial probabilities.
        Input:
            Data: a (nxt) numpy array of n samples, each t unit long. The data must have a specific 
                form to be read in where each possible emission is enumerated starting from 0 
                (called encoded data).
            encoded: a boolean representing if the data is encoded. If not, a naive EncodingScheme will be used.
        
   
        """
        
        self._fit_init(data, encoded)
        list_series_length = range(1, self.len_trials)

        for encoded_series in data:
            
            self.init_probs_estimate[ encoded_series[0] ] += 1
            for j in list_series_length:
                self.trans_probs_estimate[ encoded_series[j-1], encoded_series[j] ] += 1 
        
            self.number_of_series += 1
        self.init_probs_estimate = self._normalize( self.init_probs_estimate )
        self.trans_probs_estimate = self._normalize( self.trans_probs_estimate )
        
    def sample(self, n=1):
        """
        Sample the learned model n times.
        
        """
        samples = np.empty( (n, self.len_trials) )
        for i in range(n):
            samples[i,:] = self._sample()
        return samples
        
    def _sample(self):
        sample = np.empty( (1,self.len_trials) )
        sample[0,0] = np.argmax(np.random.multinomial(1, self.init_probs_estimate )) # argmax. something like this.
        for i in range( 1, self.len_trials):
            sample[0, i] = np.argmax(np.random.multinomial( 1, self.trans_probs_estimate[ sample[0,i-1],: ] ) )
        return sample
    
    def decode_sample(self, sample):
        """return decoded samples based on the encoding scheme"""
        return self.encoding.decode( sample )

    def _normalize(self, array ):
        #normalizes the array to sum to one. The array should be semi-positive
        try:
            #2d?
            return array.astype("float")/array.sum(1)[:,None]
        except: 
            #oh, 1d
            return array.astype("float")/array.sum()
         


     
    def __sample_conditional( self, K, X):
        #K and X are a list, K is increasing positions, min(K)>0
        # TODO
        sample = np.empty( (1, self.len_trials) )
        for i,k in enumerate(K):
            substr = self._sample_conditional( X[i] )
            pass
                    
    def sample_conditional(self, k, x, negate=False):
        #Sample the process, but at position k, put x (or put NOT x).
        sample = np.empty( (1, self.len_trials) )
        negate = int(negate) #0 or 1
        sample[0,0] = np.argmax( np.random.multinomial( 1, self.init_probs_estimate ) )
        for i in range(1, k + negate):
            A = np.linalg.matrix_power( self.trans_probs_estimate, k-i )
            if not negate:
                p = self.trans_probs_estimate[ sample[0,i-1], :]*A[:, x ]
            else:
                p = self.trans_probs_estimate[ sample[0,i-1], :]*(1-A[:, x ])

            p = self._normalize(p)
            sample[0, i] = np.argmax( np.random.multinomial( 1, p ) )
        
        if not negate:
            sample[0, k] = x
            
        
        for i in range(k+ 1, self.len_trials):
                        sample[0, i] = np.argmax(np.random.multinomial( 1, self.trans_probs_estimate[ sample[0,i-1],: ] ) )
        return sample
        
            

    def _fit_init(self,data, encoded):    
        
        if not encoded:
            if not self.encoding:
                self.encoding = encoding.EncodingScheme()
            data = self.encoding.encode(data)

            
        self.number_of_series = 0
        self.data = data
        self.unique_elements = np.arange( len( self.encoding.unique_bins)    )[None, :]
        self.len_trials = self.encoding.series_length
        
        #self.n_trials, self.len_trials = data #iterators do not have a defined shape. This might have to be done on the fly.
        self.init_probs_estimate = np.zeros( self.unique_elements.shape[1], dtype="int" )
        self.trans_probs_estimate = np.zeros( (self.unique_elements.shape[1], self.unique_elements.shape[1]), dtype="int" )
            
            
            
        
            
        