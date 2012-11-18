#supervised PCA according to Supervised Principal Compontent Anaysis by Ghodsi et al. 2010

import numpy as np
from scipy import linalg

from ..utils.arpack import eigsh
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import KernelCenterer, scale
from ..metrics.pairwise import pairwise_kernels


from time import clock



class Supervised_PCA(BaseEstimator, TransformerMixin):
    """Supervised Principal component analysis (SPCA)

    Non-linear dimensionality reduction through the use of kernels.

    Parameters
    ----------
    n_components: int or None
        Number of components. If None, all non-zero components are kept.

    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "precomputed"
        Kernel.
        Default: "linear"

    degree : int, optional
        Degree for poly, rbf and sigmoid kernels.
        Default: 3.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels.
        Default: 1/n_features.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.


    eigen_solver: string ['auto'|'dense'|'arpack']
        Select eigensolver to use.  If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.

    tol: float
        convergence tolerance for arpack.
        Default: 0 (optimal value will be chosen by arpack)

    max_iter : int
        maximum number of iterations for arpack
        Default: None (optimal value will be chosen by arpack)

    Attributes
    ----------

    `lambdas_`, `alphas_`:
        Eigenvalues and eigenvectors of the centered kernel matrix


    """
    
    def __init__(self, n_components=None, kernel="linear", gamma=0, degree=3,
                 coef0=1, alpha=1.0, fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None):
                 
                 
        self.n_components = n_components
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.centerer = KernelCenterer()
        
        
    def transform(self, X):
	"""
	Returns a new X, X_trans, based on previous self.fit() estimates
	"""
	return X.dot( self.alphas_ )  
        
    
    def fit(self,X,Y):
        self._fit(X,Y)
        return 
        
    def fit_transform( self, X, Y):
    
    
        self.fit( X,Y)
        return self._transform()
        
    def _transform(self):
        
        return self.X_fit.dot(self.alphas_)

        
    def _fit(self, X, Y):
        #find kenerl matrix of Y
	K = self.centerer.fit_transform(self._get_kernel(Y))
        #scale X
        X_scale = scale(X)
        
        
        if self.n_components is None:
            n_components = K.shape[0]
        else:
            n_components = min(K.shape[0], self.n_components)
        
        #compute eigenvalues of X^TKX
        
        M = (X.T).dot(K).dot(X)
	if self.eigen_solver == 'auto':
            if M.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(
                M, eigvals=(M.shape[0] - n_components, M.shape[0] - 1))
        elif eigen_solver == 'arpack':
            self.lambdas_, self.alphas_ = eigsh(M, n_components,
                                                which="LA",
                                                tol=self.tol)
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]
        
        #remove the zero/negative eigenvalues
        self.alphas_ = self.alphas_[:, self.lambdas_ > 0 ]
        self.lambdas_ = self.lambdas_[ self.lambdas_ > 0 ]
        print self.alphas_.shape
        
        self.X_fit = X;
        
        
    def _get_kernel(self, X, Y=None):
        params = {"gamma": self.gamma,
                  "degree": self.degree,
                  "coef0": self.coef0}
        try:
            return pairwise_kernels(X, Y, metric=self.kernel,
                                    filter_params=True,  n_jobs = -1, **params)
        except AttributeError:
            raise ValueError("%s is not a valid kernel. Valid kernels are: "
                             "rbf, poly, sigmoid, linear and precomputed."
                             % self.kernel)
        
    
        
         
class Kernel_Supervised_PCA(    BaseEstimator, TransformerMixin):    

    """Kernel Supervised Principal component analysis (SPCA)

    Non-linear dimensionality reduction through the use of kernels.

    Parameters
    ----------
    n_components: int or None
        Number of components. If None, all non-zero components are kept.

    x||ykernel: "linear" | "poly" | "rbf" | "sigmoid" | "precomputed"
        Kernel.
        Default: "linear"

    degree : int, optional
        Degree for poly, rbf and sigmoid kernels.
        Default: 3.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels.
        Default: 1/n_features.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.


    eigen_solver: string ['auto'|'dense'|'arpack']
        Select eigensolver to use.  If n_components is much less than
        the number of training samples, arpack may be more efficient
        than the dense eigensolver.

    tol: float
        convergence tolerance for arpack.
        Default: 0 (optimal value will be chosen by arpack)

    max_iter : int
        maximum number of iterations for arpack
        Default: None (optimal value will be chosen by arpack)

    Attributes
    ----------

    `lambdas_`, `alphas_`:
        Eigenvalues and eigenvectors of the centered kernel matrix


    """
    
    def __init__(self, n_components=None, xkernel={'kernel': "linear", 'gamma':0, 'degree':3,
                 'coef0':1}, ykernel = {'kernel': "linear", 'gamma':0, 'degree':3,
                 'coef0':1},  fit_inverse_transform=False,
                 eigen_solver='auto', tol=0, max_iter=None):
                 
                 
        self.n_components = n_components
        self.xkernel = xkernel
        self.ykernel = ykernel
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.centerer = KernelCenterer()
        
        
    def transform(self, X):
        """
        Returns a new X, X_trans, based on previous self.fit() estimates
        """
        K = self._get_kernel(self.X_fit, self.xkernel, X )
        return K.T.dot( self.alphas_ )  
            
    
    def fit(self,X,Y):
        self._fit(X,Y)
        return 
        
    def fit_transform( self, X, Y):
    
    
        self.fit( X,Y)
        return self._transform()
        
    def _transform(self):
        
        return self.Kx_fit.dot(self.alphas_)

        
    def _fit(self, X, Y):
        #find kenerl matrix of Y
        Ky = self.centerer.fit_transform(self._get_kernel(Y), self.ykernel)
        Kx = self.centerer.fit_transform( self._get_kernel(X), self.xkernel)
        

        
        if self.n_components is None:
            n_components = Ky.shape[0]
        else:
            n_components = min(Ky.shape[0], self.n_components)
        
        #compute eigenvalues of X^TKX
        
        M = (Kx).dot(Ky).dot(Kx)
	if self.eigen_solver == 'auto':
            if M.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(
                M, Kx, eigvals=(M.shape[0] - n_components, M.shape[0] - 1))
        elif eigen_solver == 'arpack':
            self.lambdas_, self.alphas_ = eigsh(M, Kx, n_components,
                                                which="LA",
                                                tol=self.tol)
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]
        
        #remove the zero/negative eigenvalues
        self.alphas_ = self.alphas_[:, self.lambdas_ > 0 ]
        self.lambdas_ = self.lambdas_[ self.lambdas_ > 0 ]
        
        self.X_fit = X;
        self.Kx_fit = Kx;
        
    def _get_kernel(self, X, params, Y=None):
        try:
            return pairwise_kernels(X, Y, metric=params['kernel'],
                                     n_jobs = -1, **params)
        except AttributeError:
            raise ValueError("%s is not a valid kernel. Valid kernels are: "
                             "rbf, poly, sigmoid, linear and precomputed."
                             % params['kernel'])
        
    
