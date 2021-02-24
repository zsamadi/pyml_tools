# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:53:53 2021

@author: zayn

This developement of svm algorithm is based on smo algorithm developed by
John C. Platt's  
"Sequential Minimal Optimization:
A Fast Algorithm for Training Support Vector Machines""


"""

import numpy as np
import matplotlib.pyplot as plt

eps=1e-3
class svmal:
    def __init__(self,C=1,ktype='linear', gamma=1):
        self.gamma=gamma
        self.ktype=ktype
        self.C=float(C)
        
    def plotData(self, X, y):
        X_pos=X[y>0]
        X_neg=X[y<1]
        fig, ax=plt.subplots()
        ax.plot(X_pos[:,0], X_pos[:,1],'bx', markersize=2.5, label='Positive')
        ax.plot(X_neg[:,0], X_neg[:,1],'yo', markersize=2.5, label='Negetive')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],loc='lower left', shadow=True, fontsize='medium')
        return ax
    def visualizeBoundryLinear(self,ax, w, b, X):
        x0=np.linspace(min(X[:,0]), max(X[:,0]), 100)
        x1=-(b+w[0]*x0)/w[1]
        ax.plot(x0, x1,'r-')
        return 0
    def visualizeBoundry(self, ax, alpha_vec, b,X,y, ktype, gamma):

        self.Xtrain=X
        self.ytrain=y
        self.ktype=ktype
        self.alphas=alpha_vec
        self.b=b
        self.gamma=gamma

        x0len=100
        x1len=110        
        xp0=np.linspace(min(X[:,0]), max(X[:,0]), x0len)
        xp1=np.linspace(min(X[:,1]), max(X[:,1]), x1len)
        axp0=np.kron(xp0, np.ones(len(xp1)))
        axp1=np.kron(np.ones(len(xp0)), xp1)

        xc=np.zeros([x0len*x1len,2])
        
        xc[:,0]=axp0
        xc[:,1]=axp1
        Cpr=self.predict(xc)
        Cpr.shape=(x0len, x1len)
                
        Xm1, Xm2=np.meshgrid(xp0,xp1)
        # plt.contourf(Xm1, Xm2, Cpr, 10, alpha=.75, cmap='jet')
        ax.contour(Xm1, Xm2, Cpr.T, 1)
    def predict(self,xc):
        X=self.Xtrain
        kmat=self.kernel(xc, X)
        alpha_vec=self.alphas
        b=self.b
        y=self.ytrain
        
        yalpha=y*alpha_vec
        p1=kmat.T@yalpha
        pout=p1+b

        return pout
    def kernel(self, x0, x1):
        if self.ktype=='linear':
            k_go=x0@x1.T
        else:
            kmat=np.exp(-self.gamma)
            x02=np.sum(x0*x0, axis=1)
            x12=np.sum(x1*x1, axis=1)
            x0x1=x0@x1.T
            K=(x02+(x12-2*x0x1).T)
            k_go=kmat**K
        return k_go
            
    def svmz(self,X, y, tol, max_passes):
        
                

        b,alpha_vec=self.som_main_routine(X,y,tol, max_passes)
        self.b=b
        self.alphas=alpha_vec
        self.Xtrain=X
        self.ytrain=y
        
        # save the model
        return  self




    def som_main_routine(self, X,y,tol=0.005, max_passes=10): 
        
        Kmat=self.kernel(X, X) 
        C=self.C
        num_changed=0
        examine_all=True
        alpha_vec=0.0*np.random.rand(len(y))
        b=0.0

        passes=0
        yalpha=y*alpha_vec
        u=Kmat@yalpha+b
        self.cachE=u-y
        


        while(passes<max_passes):
            num_changed=0
            if (examine_all):
                for i1 in range(alpha_vec.size):
                    if_changed,alpha_vec, b= self.examine_example(i1, X, y, b, alpha_vec, C, Kmat)
                    
                    num_changed+=if_changed
            else:
                alpha_idx_n0C=np.where((alpha_vec!=0) * (alpha_vec!=C))[0]

                for i1 in alpha_idx_n0C:
                    if_changed,alpha_vec, b= self.examine_example(i1, X, y, b, alpha_vec, C, Kmat)
                    
                    num_changed+=if_changed
            if num_changed==0:
                passes +=1
            else:
                passes=0
                    
                        
            if (examine_all):
                examine_all=False
            elif (num_changed==0):
                examine_all=True
        return  b,alpha_vec
    
    def examine_example(self, i1,X,y,b, \
                    alpha_vec,C, Kmat,tol=0.005):
        y1=y[i1]
        alpha1=alpha_vec[i1]

        E1=self.cachE[i1]
        r1=E1*y1


        alpha_vec_n0=alpha_vec[alpha_vec!=0]
        alpha_vec_n0C=alpha_vec_n0[alpha_vec_n0!=C]
        num_n0C=alpha_vec_n0C.size
        
            
        
        
        if (r1<-tol and alpha1<C) or (r1>tol and alpha1>0):
            if num_n0C>1:
                i0=self.i0_heuristic(X,y,b, \
                                      alpha_vec,i1, C, Kmat,tol)
                    
                pos_step, alpha_vec, b=self.take_pos_step(i0, i1,X, y, C,alpha_vec,b,  Kmat,tol)
                if pos_step==1:
                    return 1, alpha_vec,b

            for i0 in range(alpha_vec.size):
                pos_step, alpha_vec, b=self.take_pos_step(i0, i1,X, y, C,alpha_vec,b,  Kmat,tol)
                if pos_step==1:
                    return 1, alpha_vec, b
        return 0, alpha_vec, b

    def i0_heuristic(self, X,y,b,alpha_vec,i1, C, Kmat,tol=0.005):
        E1=self.cachE[i1]
        ED=abs(self.cachE-E1)
        maxi0=i1
        while(maxi0==i1):
            maxi0=np.argmax(ED)
            ED[maxi0]=-1
        

        return maxi0    
        
    def take_pos_step(self,i0, i1, X, y, C,alpha_vec,b,  Kmat,tol): 
        if (i0==i1):
            return 0, alpha_vec,b
        y0=y[i0]
        y1=y[i1]

        alpha0=alpha_vec[i0]
        alpha1=alpha_vec[i1]

        
        
        
        if y0==y1:
            Lb=max(alpha1+alpha0-C, 0)
            Hb=min(C, alpha1+alpha0)
            s=1
        else:
            Lb=max(alpha1-alpha0, 0)
            Hb=min(C, C-alpha0+alpha1)
            s=-1
        if (Lb==Hb):
            return 0, alpha_vec,b
        
        k00=Kmat[i0,i0]
        k01=Kmat[i0,i1]
        k11=Kmat[i1,i1]
        
        E0=self.cachE[i0]
        E1=self.cachE[i1]
        
        eta=k00+k11-2* k01
        if eta>0:
            a1=alpha1+y1*(E0-E1)/eta
            if a1<Lb:
                a1=Lb
            elif a1>Hb:
                a1=Hb
        else:
            return 0, alpha_vec,b
        if np.abs(a1-alpha1)<eps*(a1+alpha1+eps):
            return 0, alpha_vec, b
        
        a0=alpha0+s*(alpha1-a1)
        b=self.get_b(E0,E1,y0,a0,alpha0, \
                     k00,k01,k11,y1,a1,alpha1,b, C)
        alpha_vec[i0]=a0
        alpha_vec[i1]=a1

        yalpha=y*alpha_vec
        u=Kmat@yalpha+b
        

        self.cachE=u-y


            
        return 1, alpha_vec, b
    
    def get_b(self, E0,E1,y0,alpha0_new,alpha0, \
          k00,k01,k11,y1,alpha1_new_clipped,alpha1,b, C):
        b0v=0.0
        b1v=0.0
        alpha0dt=float((alpha0_new-alpha0))
        alpha1dt=float((alpha1_new_clipped-alpha1))

        alpha0d=alpha0dt if y0>0 else -alpha0dt
        alpha1d=alpha1dt if y1>0 else -alpha1dt
        
        b0= b-(E0+alpha0d*k00+alpha1d*k01)
        b1=b-(E1+alpha0d*k01+alpha1d*k11)      
                    
        if alpha0_new>0 and  alpha0_new<C:
            b0v=1
            b=b0
        if alpha1_new_clipped>0 and  alpha1_new_clipped<C:
            b1v=1
            b=b1
        if (b0v+b1v)==0:
            b=(b0+b1)/2
        return b
    
    def score(self, Xval, yval):
        yp=self.predict(Xval) 
        aa=sum((yp>0)==yval)/len(yval)
        return aa
    

