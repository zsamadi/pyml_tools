
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:07:50 2021

@author: zayn
"""

"""
Linear regression implementation.

"""
import random
import numpy as np
import matplotlib.pyplot as plt
import time

epsv = np.finfo(float).eps
class nncost_fn(object):
    def cost_fn(a,y):
        J=sum(y*np.log(a)+(1-y)*np.log(1-a))
        Jr=np.nan_to_num(J)
        Js=-sum(Jr)
        return Js
class nnet(object):
    """ implementation of multi mariable Linear Regression, 
        with results derived using steepest descent and 
        theoretical normal equations.
    """

    def __init__(self, sizes, cost_fn=nncost_fn):
        """Create a naive bayes classifier.
        :param training_data: training feature data.
        :param use_norm: Whether to use normalizing when calculating linear regression.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.J_h=0

        self.flatten=lambda t: [item for sublist in t for item in sublist]
        self.cost_fn=cost_fn
    def plotData(self, training_data):

        
        Xpi=random.sample(training_data, 100)

        
        
        Xp=[Xpi[k][0] for k in range(100)]
        Xp=((np.array(Xp)).reshape(784,100))
         

        
        # Xp, y=np.random.choice(training_data, 100, replace=False)
        
        
        Xpn=Xp.reshape(2800,28)
        Xpf=np.zeros([280,280])
        for ii in range(10):
            Xpt=Xpn[ii*280:(ii+1)*280,:]
            Xpf[:280,ii*28:(ii+1)*28]=Xpt
        
        fig, ax = plt.subplots()
        ax.imshow(Xpf, extent=[0, 1, 0, 1])

        return ax
    def sigmoid(self, X):
        gden=1+np.exp(-X)
        z=1.0/gden
 
        return z
    

    def nnGrad(self, training_data, rlambda=1):
        

        Xt=[training_data[ii][0] for ii in range(len(training_data))]
        Xt=np.array(Xt).reshape(len(training_data),len((training_data[0][0])))
        yt=[training_data[ii][1] for ii in range(len(training_data))]
        yt=np.array(yt).reshape(len(training_data),len((training_data[0][1])))

        ao, zo=self.feedforward(Xt.T)
        Do, delo=self.bward_prop(yt.T,ao,zo)


            



        
        

            
        
        # err=y.T@np.log(sig_preiction)+(1-y).T@np.log(1-sig_preiction)
        # J1=-err/m

        # reg_lambda=rlambda*(theta[1:]).T@(theta[1:])/2
        # J2=reg_lambda/m
        # J=J1+J2
        # grad=np.zeros(theta.shape)
        # grad[0]=sum(sig_preiction-y)/m 
        # Xm=X[:,1:];
        # grad[1:]=Xm.T@(sig_preiction-y)/m+ rlambda/m*theta[1:]
        # grad=grad/np.linalg.norm(grad)
        
        return Do, delo
    def feedforward(self,a):
        zo=[]
        ao=[a]
        for ii in range(len(self.biases)):
            z=self.weights[ii]@a+self.biases[ii]
            a=self.sigmoid(z)
            zo=zo+[z]
            ao=ao+[a]
        return ao, zo
    def bward_prop(self,y,at,zt):
        # Quadratic cost_fn Function
        # delta=(at[-1]-y)*self.sigmoidGradient(zt[-1]) 
        # Cross Entropy cost_fn Function
        delta=(at[-1]-y) 

        
        Do=[delta@(at[-2]).T]
        delo=[sum(delta.T)]

            
        for ii in range(2, self.num_layers):
            z=zt[-ii]
            a=at[-ii-1]
            delta=(self.weights[-ii+1].T@delta)*self.sigmoidGradient(z)
            D=[delta@a.T]
            delo=[sum(delta.T)]+delo
            Do=D+Do
            
        return  Do, delo
    
            
    def nnGraDescent(self,training_data,  num_passes, batch_size,alpha,test_data=None,rlambda=0):
    
        # Initialize some useful values

        if test_data:
            test_data=list(test_data)
            
        
        m=len(training_data)
        
        # grad_history=[]
        J_h=np.zeros(num_passes)
        regl_par=1-alpha*rlambda/m
        alpha_batch=alpha/batch_size

        for iter in range(num_passes):
            t = time.time()
            random.shuffle(training_data)
            
            training_data_batch=[training_data[k:k+batch_size] 
                                 for k in range(0,m,batch_size)]
                
            for itraining_data_batch in training_data_batch:
                D, delta=self.nnGrad(itraining_data_batch, rlambda)
                self.weights=[regl_par*w0-alpha_batch*Di for w0, Di in zip(self.weights, D)]
                self.biases=[b0-alpha_batch*deli.reshape(len(deli),1) for b0, deli in zip(self.biases, delta)]
            
            a, y=self.comp_out(training_data)
            regw_cost=0
            if rlambda>0:
                regw_cost=rlambda/2*sum([sum(sum(w0*w0)) for w0 in self.weights])
                
            J_ht=(self.cost_fn).cost_fn(a,y)+regw_cost                
            J_h[iter]=J_ht/m
            
            if test_data:
                yp=self.evaluate(test_data)
                print('iteration {}:{}%'.format(iter,yp))
            else:
                print('iteration {} complete'.format(iter))
                
            # print('Elapsed time is {0:.2f}'.format(time.time()-t))


            self.J_h=J_h
                
            
            
            # theta=theta/np.linalg.norm(theta)
            # grad_history=np.append(grad_history, grad, axis=1)
            
        
        return None

    
    def evaluate(self, test_data):

        a, yt=self.comp_out(test_data)          

        eval_out=np.argmax(a, axis=1)
        precision=sum(eval_out==yt)/len(yt)
        

        return '{0:.2f}'.format(precision*100)

    def sigmoidGradient(self, X):
        X[abs(X)<epsv]=epsv
        gden=1+np.exp(-X)
        z=1.0/gden
        zg=z*(1-z)
 
        return zg
    def comp_out(self, test_data):
        Xt=[test_data[ii][0] for ii in range(len(test_data))]
        Xt=np.array(Xt).reshape(len(test_data),len((test_data[0][0])))
        yt=[test_data[ii][1] for ii in range(len(test_data))]
        if isinstance(test_data[0][1], (list, np.ndarray)):
            yt=np.array(yt).reshape(len(test_data),len((test_data[0][1])))
        else:
            yt=np.array(yt)
                
        
        a=Xt.T
        for ii in range(len(self.weights)):
            z=self.weights[ii]@a+self.biases[ii]
            a=self.sigmoid(z) 
        return a.T, yt





         
         
    
    
    

