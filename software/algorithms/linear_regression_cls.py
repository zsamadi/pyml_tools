# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:07:50 2021

@author: zayn
"""

"""
Linear regression implementation.

"""

import numpy as np


class LReg:
    """ implementation of multi variable Linear Regression, 
        with results derived using steepest descent and 
        theoretical normal equations.
    """

    def __init__(self, training_data=[], use_norm=True):
        """Create a linear regression classifier.
        :param training_data: training feature data.
        :param use_norm: Whether to use normalizing when calculating linear regression.
        """
        if use_norm:  
            self.normed_training_data, self.mean, self.std = self.featureNormalize(training_data)
        else:
             self.normed_training_data= training_data
             self.mean=[]
             self.std=[]
             
        self.use_norm=use_norm

    def computeCost(self, X, y, theta):
        m=len(y)
        preiction=X@theta
        err_abs=(preiction-y).T@(preiction-y)
        J=err_abs/(2*m)
        return J
    def gradientDescent(self, X, y, theta, alpha, num_iters):
    
        # Initialize some useful values
        m=len(y)
        J_history=np.zeros([num_iters, 1])
        for iter in range(num_iters):
            J_history[iter]=self.computeCost(X, y, theta) #Save the cost J in every iteration 
            preiction=X@theta
            err=(preiction-y)
            update=X.T@err
            theta=theta-alpha/m*update
        
        return theta, J_history
    def featureNormalize(self, X):
        X=np.asarray(X)
        XT=X.T;
        XnT=np.zeros(XT.shape)
        xmeanv=np.zeros(XT.shape[0])
        xstdv=np.zeros(XT.shape[0])
        for ii in range(XT.shape[0]):
            xmean=np.mean(XT[ii,:]);
            xstd=np.std(XT[ii,:])
            Xub=XT[ii,:]-xmean;
            XnT[ii,:]=Xub/xstd
            xmeanv[ii]=xmean
            xstdv[ii]=xstd
        Xn=XnT.T
   
        return Xn, xmeanv, xstdv

    def normalEqn(self, X, y):
        XT=X.T
        XTX=XT@X
        pinv=np.linalg.pinv(XTX)
        XTy=XT@y
        theta=pinv@XTy
        return theta
