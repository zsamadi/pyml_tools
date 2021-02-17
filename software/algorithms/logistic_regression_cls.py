# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:38:07 2021

@author: zayn
"""

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

class lgreg:
    """ implementation of multi variable Logistic Regression, 
        with results derived using steepest descent 
    """

    def __init__(self, training_data=[], use_norm=True):
        """Create a logistic regression classifier.
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
    def plotData(self, X, y, xlabel='xlabel', ylabel='ylabel'):
        Xpos=X[y>0.5]
        Xneg=X[y<0.5]
        fig1, ax1 = plt.subplots()
        ax1.plot(Xpos[:,0],Xpos[:,1],'k+', label='Positive')
        ax1.plot(Xneg[:,0],Xneg[:,1],'yo', label='Negetive')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::-1], labels[::-1])
        return ax1
    def sigmoid(self, X):
        gden=1+np.exp(-X)
        z=1.0/gden
 
        return z
    

    def CostFunction(self, X, y, theta, rlambda=0):
        m=len(y)
        preiction=X@theta
        sig_preiction=self.sigmoid(preiction)
        
        err=y*np.log(sig_preiction)+(1-y)*np.log(1-sig_preiction)
        J1=-sum(err)/m

        reg_lambda=rlambda*sum(theta[1:]**2)/2
        J2=reg_lambda/m
        J=J1+J2
        grad=np.zeros(theta.shape)
        grad[0]=sum(sig_preiction-y)/m 
        Xm=X[:,1:];
        grad[1:]=Xm.T@(sig_preiction-y)/m+ rlambda/m*theta[1:]
        
        return J, grad
    def gradientDescent(self, X, y, theta, alpha, num_iters, rlambda=0):
    
        # Initialize cost function history
        J_history=np.zeros(num_iters)

        for iter in range(num_iters):
            J_history[iter], grad=self.CostFunction(X, y, theta, rlambda) #Save the cost J in every iteration 
            theta=theta-alpha*grad
        
        return theta, J_history
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
            XnT[ii,:]=Xub/xstd
            xmeanv[ii,0]=xmean
            xstdv[ii,0]=xstd
        Xn=XnT.T
   
        return Xn, xmeanv, xstdv
    def predict(self, X, theta, softp=False):
        preiction=X@theta
        sig_preiction=self.sigmoid(preiction)
        predico=sig_preiction if softp else np.round(sig_preiction)

        return predico
    
    def mapFeature(self, X1, X2):
        m=len(X1)
        degree=6
        num_par=sum(range(degree+2)) 
        out=np.zeros([m,num_par])
        indx=0
        for ii  in range(degree+1):
            for jj in range(ii+1):
                out[:, indx] = (X1**(ii-jj))*(X2**jj)
                indx+=1


        return out
