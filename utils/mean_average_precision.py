#mean average precision
"""
% This function computes the average precision of predicated values. The
% average precision is hella-confusing at first glace. Here's what Kaggle
% has to say:
%   
%   The true scores are sorted (descending) according to the order of the 
%   submission (only the order of the submission matters). In each row, 
%   we then compute the cumulative (from the top up to that row) 
%   "True Scores Ordered by Submission" divided by the cumulative "True 
%   Scores Ordered By True Scores", where that quotient is called the 
%   precision at row n. The final score is the average of the precision 
%   at row n (over all n).
%
%
%  Ok, so say the true scores, sorted, are 3, 2.3, 1.6. And I predicted 
% the order 3, 1.6, 2.3. Then the average prec. is mean( 3/3,
% (3+1.6)/(3+2.3), (3+1.6 + 2.3)/(3+ 2.3 + 1.6) ) = .96 something. 
%
%
"""
import numpy as np

def MAP( true_scores, predictive_scores):
    true_values_sorted = true_scores.copy()
    true_values_sorted = true_values_sorted[ np.argsort( -true_values_sorted ) ]
    
    ix = np.argsort( -predictive_scores )
    
    true_values_sorted_by_prediction = true_scores[ix]
    
    score = np.mean( true_values_sorted_by_prediction.cumsum()/ true_values_sorted.cumsum() )
    return score
    