
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:07:50 2021

@author: zayn
"""

"""
Linear regression implementation.

"""

import numpy as np
import matplotlib.pyplot as plt

epsv = np.finfo(float).eps
class lfit:
    """ implementation of multi mariable Linear Regression, 
        with results derived using steepest descent and 
        theoretical normal equations.
    """

    def __init__(self, training_data=[], use_norm=True):
        """Create a naive bayes classifier.
        :param training_data: training feature data.
        :param use_norm: Whether to use normalizing when calculating linear regression.
        """

        self.normed_training_data, self.mean, self.std = self.featureNormalize(training_data)
        self.use_norm=use_norm
        self.flatten=lambda t: [item for sublist in t for item in sublist]

    def sigmoid(self, X):
        gden=1+np.exp(-X)
        z=1.0/gden
 
        return z
    

    def linearRegCostFunction(self, X, y, theta,rlambda=0):
        m=y.size
        XTheta=X @ theta
        err=XTheta-y
        err_norm=err.T@err
        theta_1=theta[1:]
        tetha1_norm=theta_1.T@theta_1
        J=1/(2*m)*(err_norm+rlambda*tetha1_norm)
        grad=1/m*X.T @ err

        grad[1:]+=rlambda/m*theta[1:]
        
        # grad=grad/np.std(grad)
        
        return J,grad
    def trainLinearReg(self, X, y,  num_iters, alpha,rlambda=1):
        n=X.shape[1]
        theta=np.zeros([n,1])
    
        
    
        # Initialize some useful values
        J_history=np.zeros([num_iters, 1])
        
        # grad_history=[]

        for iter in range(num_iters):
            J_history[iter], grad=self.linearRegCostFunction(X, y,theta, rlambda)
             
             
            theta=theta-alpha*grad
            # theta=theta/np.linalg.norm(theta)
            # grad_history=np.append(grad_history, grad, axis=1)
            
        
        return theta, J_history, grad
    def learningCurve(self, X, y,  X_val, y_val,rlambda=1):
        n=X.shape[1]
        theta=np.zeros([n,1])
        num_iters=400
        alpha=0.03
        m=y.size
        error_train=np.zeros(m)
        error_val=np.zeros(m)
        
        for ii in range(m):
            X_tin=X[:ii+1,:]
            y_tin=y[:ii+1,:]
            
            theta, J_history, grad=self.trainLinearReg(X_tin, y_tin,  num_iters, alpha,rlambda)
            J_val, g_val=self.linearRegCostFunction(X_tin, y_tin,theta, rlambda)
            error_train[ii]=J_val
            J_val, g_val=self.linearRegCostFunction(X_val, y_val,theta, rlambda)
            error_val[ii]=J_val
            

            
        
        return error_train, error_val  
    
    def featureNormalize(self, X):
        X=np.asarray(X)
        XT=X.T;
        XnT=np.zeros(XT.shape)
        xmeanv=np.zeros([XT.shape[0],1])
        xstdv=np.zeros([XT.shape[0],1])
        for ii in range(XT.shape[0]):
            xmean=np.mean(XT[ii,:]);
            xstd=np.std(XT[ii,:])
            Xub=XT[ii,:]-xmean;
            XnT[ii,:]=Xub/xstd if xstd>0 else Xub
            xmeanv[ii,0]=xmean
            xstdv[ii,0]=xstd
        Xn=XnT.T
   
        return Xn, xmeanv, xstdv
    def polyfeatures(self, X,p):
        Xp=np.zeros([X.shape[0],p])
        for ii in range(p):
            Xp[:,ii]=(X**(ii+1))[:,0]
        return Xp
    def postNormalize(self, X, mu, sigma):
        Xn=np.zeros(X.shape)
        for ii in range(mu.size):
            Xn[:,ii]=(X[:,ii]-mu[ii])/sigma[ii]
        return Xn
    

    def plotFit(self, min_x, max_x, mu, sigma, theta, p):
        X=np.arange(min_x, max_x, 0.05)
        X.shape=(X.size, 1)
        Xp=self.polyfeatures(X,p)
        Xpn=np.zeros(Xp.shape)
        for ii in range(mu.size):
            Xpn[:,ii]=(Xp[:,ii]-mu[ii])/sigma[ii]
        Xpne=np.append(np.ones([Xpn.shape[0],1]),Xpn, axis=1)
        y=Xpne@theta

        return Xpn[:,0], y
    def validationCurve(self,X, y, X_val, y_val):
        num_iters=400
        alpha=0.03
        rlambda_vec=np.array([0,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3,10])
        error_train=np.zeros(rlambda_vec.shape)
        error_val=np.zeros(rlambda_vec.shape)

        for ii in  range(rlambda_vec.size):
            rlambda=rlambda_vec[ii]
            theta, J_history, grad=self.trainLinearReg(X, y,  num_iters, alpha,rlambda)
            J_val, g_val=self.linearRegCostFunction(X, y,theta, rlambda=0)
            error_train[ii]=J_val
            J_val, g_val=self.linearRegCostFunction(X_val, y_val,theta, rlambda=0)
            error_val[ii]=J_val
        return rlambda_vec, error_train, error_val
            

    