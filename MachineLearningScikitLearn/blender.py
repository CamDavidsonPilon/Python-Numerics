import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import ShuffleSplit
from time import clock
import pp

class Blender( object):
    """
        This class implements a linear blend of different models.
    
    
        methods:
            fit( data, response, dict_of_additional_variables )
            add_model( model, name)
            predict( new_data, dict_of_additional_variables )
        
        
        attributes:
            coefs_
            
    """
    
    
    def __init__( self, blender = LinearRegression(), training_fraction = 0.8, verbose = False):
            self.blender = blender
            self.training_fraction = training_fraction
            self.verbose = verbose
            self.models = dict()
            self._n_models = 0
    
    
    def add_model(self, model, name=None):
        """ 
            model: a sklearn model that exposes the methods fit & predict. 
            name: a name to specify the model, eg "ElNet500alpha"
        
        """
	self._n_models +=1
        if not name:
            name = "%d"%( self._n_models )
        self.models[name] = model
        return
        
    def del_model(self, name ):
	try:	
	   del self.models[name]
	except KeyError:
	   print "Model %s not in blender."%name

	return


    def split_arrays(self, n,  test_fraction = 0.1 ):
	
	
	shfSplt = ShuffleSplit( n=n, n_iterations=1, test_size = test_fraction)
	train_ix, test_ix = shfSplt.__iter__().next()
	return train_ix, test_ix

 	
 
    def fit(self, data, response, dict_of_additional_variables={}):
        """
            data: the data matix, shape (n,d)
            response: the response vector (n,)
            dict_of_additional_variables:
                a dictionary with the keys the model names (optional to include), and the items are of the form:
                        {"train":[ items to be included in training], "test":[items to be included in testing] }
        """
        
        #split the data to held-in and held-out.
	train_ix, blend_ix = self.split_arrays( data.shape[0], test_fraction = 1- self.training_fraction )
	training_data, blend_data, training_response, blend_response = data[train_ix], data[blend_ix], response[train_ix], response[blend_ix]	

 
        X = np.zeros( (blend_response.shape[0], len( self.models ) ) )

        if self.verbose:
            print "Shape of training data vs blending data: ", training_data.shape, blend_data.shape 
        #train the models
	
	
	#try some parrallel
	ncpus = max( len( self.models ), 32 )
	job_server = pp.Server( ncpus, ppservers = () )
	jobs = dict()
	to_import = ("import numpy as np", "sklearn", "time", "from localRegression import *", "from sklearn.linear_model import sparse", "from sklearn.utils import atleast2d_or_csc") 
        for name, model in sorted( self.models.iteritems() ):
		
		try:
			fitargs = [ training_data, training_response] + [ array[train_ix] for array in dict_of_additional_variables[name ]] 
			predictargs = [ blend_data ] + [ array[blend_ix] for array in dict_of_additional_variables[name] ]
		except KeyError:
			fitargs = [ training_data , training_response]
			predictargs = [ blend_data ]
                
		jobs[name] = job_server.submit( pp_run,(model, name, self.verbose, fitargs, predictargs), (), to_import )

	    	if self.verbose:
			print "Model %s sent to cpu."%name
        
	i = 0
	for name, model in sorted( self.models.iteritems() ):
	    self.models[name], X[:,i]  = jobs[name]()
	    i+=1

        if self.verbose:
            print "Fitting finished, starting blending."
        
        self.blender.fit( X, blend_response )
        self.coef_ = self.blender.coef_
        
        self._fit_training_data = training_data
        self._fit_blend_data = blend_data
        self._fit_training_response = training_response
        self._fit_blend_response = blend_response
        
        if self.verbose:
            print "Done fitting"
        job_server.destroy()    
        return self
           
    def predict( self, data, dict_of_additional_variables={}):
        
	ncpus = max( len( self.models ), 32 )
	job_server = pp.Server( ncpus, ppservers = () )
	jobs = dict()
	to_import = ("import numpy as np", "sklearn", "time", "from localRegression import *", "from sklearn.linear_model import sparse", "from sklearn.utils import atleast2d_or_csc") 
        for name, model in sorted( self.models.iteritems() ):
		try:
			predictargs = [data] +  dict_of_additional_variables[name]
		except KeyError:	
			predictargs = [ data ] 

		jobs[name] = job_server.submit( pp_predict, (model, name, self.verbose, predictargs), (), to_import)

	X = np.zeros( (data.shape[0], len( self.models ) ) )
        i = 0
        for name, model in sorted( self.models.iteritems() ):
            X[:,i] = jobs[name]()
	    i+=1	
	job_server.destroy()
        return self.blender.predict( X )
        


def pp_predict( model, name, verbose, predictargs):
	start = time.clock()
	p = model.predict( *predictargs )
	if verbose:
		print "Model %s fitted, took %.2f seconds"%(name, time.clock() - start )
	return p

def pp_run( model, name, verbose, fitargs, predictargs):
	
	start = time.clock()
	model.fit(*fitargs)
        if verbose:
		print "Model %s fitted, took %.2f seconds."%(name, time.clock() - start )
	prediction = model.predict( *predictargs ) 
	return model, prediction
