#ensemble selection

import numpy as np
        
        
def RMSE( Z, W):
    return np.sqrt( ((Z - W[:,None])**2).mean(axis=0) )

def basis(i, N):
    z = np.zeros(N)
    z[i] = 1
    return z


class EnsembleSelection( object ):
    """
    This class implements a greedy ensemble selection algorithm outlined in Ensemble Selection from Libraries of Models.
    The algorthim starts with an initial ensemble of models (if fraction_sorted_initialization > 0), and addeds models 
    sequentially until improve falls below some threshold or the max number of models are selected.
    
    verbose: 0,1 or 2. Report the current score, number of models at each iterations.
    with_replacement: all the algorithm to select models already selected.
    fraction_sorted_initialization: The fraction of the best models to initialilly include in the ensemble
    bag_selection: Perform the following bagged_selection_times. select bagged_fraction and perform the greedy algo of them.
    bagged_fraction: see above.
    max_models: the maximum number of models to include in an ensemble
    score_function: the function to minimize.
    tol: the fractional decrease in the score_function to continue selection. RMSE_{i+1}/RMSE_{i} > 1 + tol.
    fit_models: instead of giving already fitted models, this object will fit the models too.
    training_fraction: the fraction to use for training, 1-training_fraction is used as ensemble selection.
    
    methods:
        add_model( iterable_of_models ): add a collection of models to the algorithm. Must be performed before fit() is called.
            Models must be aready fitted and have a .predict() method exposed.
        fit( X, Y): perform the ensemble selection on data X and target Y
        predict( X ): return the prediction of the ensemble.
    
    """
    
    
    def __init__(self, verbose = 1, 
                 with_replacement = True, 
                 fraction_sorted_initialization = 1.0, #bayesian prior of 1/N.
                 bag_selection = 0,
                 bagged_fraction = 0.5,
                 max_models = None,
                 score_function = RMSE,
                 tol = 1e-4,
                 fit_models = False,
                 training_fraction =0.8, 
                 models = []):
        self.verbose = verbose
        self.with_replacement = with_replacement
        self.fraction_sorted_initialization = fraction_sorted_initialization
        self.bag_selection = bag_selection
        self.bagged_fraction = bagged_fraction
        self.max_models = max_models
        self.fit_models = fit_models
        self.training_fraction = training_fraction  
        
        self.score_function = score_function
        self.tol = tol
        
        if self.max_models == None:
            self.max_models = np.inf
        
        self.models= models
    
    
    def add_model(self, model ):
        """model should be an iterable"""
        self.models += [ m for m in model ]
        
        return
    
    
    def _predict( self, predictions, models_included_ ):
        return (np.dot( predictions, models_included_ )/models_included_.sum())[:,None]
    
    def _fit( self, predictions, Y, ix):
    
        n,N = predictions.shape
        #train and store the prediction results
        models_included_ = np.zeros( N )
            
        init_n_to_include = max( int( self.fraction_sorted_initialization*N), 1)
        models_included_[ np.argsort( self.individual_scores[ix] )[:init_n_to_include] ] = 1
        
        total_scores_ = np.array( [np.inf,  self.score_function( self._predict( predictions, models_included_ ) , Y)  ] )        
        
        while (models_included_.sum() < self.max_models) and ( total_scores_[-2]/total_scores_[-1] > 1 + self.tol ) :
            
            #find the best addition. 
            _scores = [ self.score_function(self._predict(predictions, models_included_ + basis(i, N)), Y) \
                                    for i in range(N) if (models_included_[i] == 0 or self.with_replacement)] 
            m = np.argmin( _scores )
            if _scores[m] < total_scores_[-1]:
                total_scores_ = np.append( total_scores_, _scores[m] )
                models_included_[m] += 1
                if self.verbose > 1:
                    print "Added model %d."%m
                    print "Current score: %.3f."%total_scores_[-1]
                    print "Current models included: ", models_included_
                    print 
            else:
                flag = True
                break
            
        if self.verbose > 0:
            if (models_included_.sum() >= self.max_models):
                print "Exited after %d iterations because number of models exceeded. %d >= self.max_models"%(models_included_.sum(), models_included_.sum() )
            elif ( total_scores_[-2]/total_scores_[-1] <= 1 + self.tol ):
                print "Exited after %d iterations because tolerence exceeded: %.8f < 1 + tol"%(models_included_.sum(), total_scores_[-2]/total_scores_[-1])
            elif flag:
                print "The (local) minimum was found after %d iterations."%(models_included_.sum())
            print "Score: %.4f"%total_scores_[-1]
        return models_included_/models_included_.sum()
        
        
    def fit(self,  X, Y):
        N = len( self.models )
        n,d = X.shape
        
        if self.fit_models:
            cutoff = int(n*self.training_fraction)
            a = np.arange(n)
            np.random.shuffle(a)
            training_data, training_target = X[ a[:cutoff] ,:], Y[ a[:cutoff] ] 
            [ m.fit( training_data, training_target) for m in self.models ]
            
            if self.verbose > 0:
                print "models trained."
            X, Y = X[ a[cutoff:], :], Y[ a[cutoff:] ]
            n,d = X.shape
        
        #train and store the prediction results
        predictions = np.zeros( (n, N) )
        for i in range(N):
            predictions[ :, i] = self.models[i].predict( X )
            
        self.individual_scores = self.score_function( predictions, Y )
        self.models_included_ = np.zeros( N )
        p = self.bagged_fraction if self.bag_selection > 0 else 1
        
        for i in range( max(1, self.bag_selection ) ):
            a = np.arange( N) 
            np.random.shuffle( a)
            ix = a[:int(p*N) ]
            models_included_ = self._fit( predictions[:, ix], Y, ix )
            self.models_included_[ix] += models_included_
        
        self.models_included_ /= self.models_included_.sum()
        self.score_ = self.score_function( self._predict( predictions, self.models_included_), Y )
        return self
        
    
    def get_params(self, deep=False):
        return self.__dict__
    
    def predict( self, X ):
            
        N = len( self.models )
        n,d = X.shape
        #train and store the prediction results
        predictions = np.zeros( (n, N) )
        for i in range(N):
            predictions[ :, i] = self.models[i].predict( X )
            
        return self._predict( predictions, self.models_included_ )        
        
