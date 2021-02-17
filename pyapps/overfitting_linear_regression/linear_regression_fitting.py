
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:35:28 2021

@author: zayn
"""
import sys
sys.path.append('../../software/algorithms/')   
    

import numpy as np
import fitting_cls as lfit
import scipy.io
import matplotlib.pyplot as plt



mat = scipy.io.loadmat('../../data/linear_fitting.mat')

X=mat["X"]
y=mat["y"]
m=y.size

X_val=mat["Xval"]
y_val=mat["yval"]

X_test=mat["Xtest"]
y_test=mat["ytest"]






    
lreg_fit=lfit.lfit()

Xn, xmeanv, xstdv=lreg_fit.featureNormalize(X)
X_valn=(X_val-xmeanv)/xstdv



fig, ax = plt.subplots()

ax.plot(Xn, y, 'rx', markersize=10, linewidth= 1.5)
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')


# X,m_sample, sig_sample=lreg_fit.featureNormalize(X)

Xe=np.append(np.ones([Xn.shape[0],1]),Xn,axis=1)

nt=Xe.shape[1]

theta=np.zeros([nt,1])

rlambda=1

J, grad=lreg_fit.linearRegCostFunction(Xe, y, theta, rlambda)

## Part 4
num_iters=400
alpha=0.03
rlambda=1

theta, J_history, grad=lreg_fit.trainLinearReg(Xe, y,  num_iters, alpha,rlambda)
y_est=Xe@theta

ax.plot(Xn, y_est, 'b-', linewidth= 1.5)   

# fig1, ax1 = plt.subplots()

# ax1.plot(np.arange(0,J_history.size), J_history, 'r-', linewidth= 1.5)
# ax1.set_xlabel('#Num of iterations')
# ax1.set_ylabel('Cost function')

## Part 5

X_vale=np.append(np.ones([X_valn.shape[0],1]),X_valn,axis=1)

error_train, error_val  =lreg_fit.learningCurve(Xe, y,  X_vale, y_val,rlambda)

fig2, ax2 = plt.subplots()

h1=ax2.plot(np.arange(0,error_train.size), error_train, 'b-', linewidth= 1.5, label='train')
h2=ax2.plot(np.arange(0,error_val.size), error_val, 'r-', linewidth= 1.5, label='Cross Validation')
handles, labels = ax2.get_legend_handles_labels()
# ax.legend()
legend = ax2.legend(handles[::-1], labels[::-1],loc='upper center', shadow=True, fontsize='x-large')

ax2.set_xlabel('#Num of training samples')
ax2.set_ylabel('estimation error')
# Part 6
p=8
Xp=lreg_fit.polyfeatures(X,p)

Xpn,Xpmean, Xpstd=lreg_fit.featureNormalize(Xp)

Xpne=np.append(np.ones([Xpn.shape[0],1]),Xpn, axis=1)
num_iters=400
alpha=0.03
rlambda=0

## Regularized polynomial linear regression
rlambda=1

theta, J_history, grad=lreg_fit.trainLinearReg(Xpne, y,  num_iters, alpha,rlambda)


X_est, yp_est=lreg_fit.plotFit(min(X), max(X), Xpmean, Xpstd, theta, p)

# fig3, ax3 = plt.subplots()

# ax3.plot(Xn, yp_est, 'b-', linewidth= 1.5)   

ax.plot(X_est, yp_est, 'g-', linewidth= 1.5)


#############################################
X_valp=lreg_fit.polyfeatures(X_val,p)

X_valpn=lreg_fit.postNormalize(X_valp, Xpmean, Xpstd)

X_valpne=np.append(np.ones([X_valpn.shape[0],1]),X_valpn, axis=1)

error_train, error_val  =lreg_fit.learningCurve(Xpne, y,  X_valpne, y_val,rlambda)

fig4, ax4 = plt.subplots()

h1=ax4.plot(np.arange(0,error_train.size), error_train, 'b-', linewidth= 1.5, label='train')
h2=ax4.plot(np.arange(0,error_val.size), error_val, 'r-', linewidth= 1.5, label='Cross Validation')
handles, labels = ax4.get_legend_handles_labels()
# ax.legend()
legend = ax4.legend(handles[::-1], labels[::-1],loc='upper center', shadow=True, fontsize='x-large')

ax4.set_xlabel('Number of training samples')
ax4.set_ylabel('estimation error')
ax4.set_title('poly lin. regresion train and valid. error curve')


################## Lambda optimization

rlambda_vec, error_train, error_val=lreg_fit.validationCurve(Xpne, y, X_valpne, y_val)

fig5, ax5 = plt.subplots()

h1=ax5.plot(rlambda_vec, error_train, 'b-', linewidth= 1.5, label='train')
h2=ax5.plot(rlambda_vec, error_val, 'r-', linewidth= 1.5, label='Cross Validation')
handles, labels = ax5.get_legend_handles_labels()
# ax.legend()
legend = ax5.legend(handles[::-1], labels[::-1],loc='upper center', shadow=True, fontsize='x-large')

ax5.set_xlabel('Number of training samples')
ax5.set_ylabel('estimation error')
ax5.set_title('train and valid. error versus lambda')

# Evaluating algorithm on test data

rlambda=3



theta, J_history, grad=lreg_fit.trainLinearReg(Xpne, y,  num_iters, alpha,rlambda)


X_testp=lreg_fit.polyfeatures(X_test,p)

X_testpn=lreg_fit.postNormalize(X_testp, Xpmean, Xpstd)

X_testpne=np.append(np.ones([X_testpn.shape[0],1]),X_testpn, axis=1)

J_test, g_test=lreg_fit.linearRegCostFunction(X_testpne, y_test,theta, rlambda=0)

X_est, yp_est=lreg_fit.plotFit(min(X_test), max(X_test), Xpmean, Xpstd, theta, p)

fig, ax = plt.subplots()

ax.plot(X_testpn[:,0], y_test, 'bx', markersize=2.5,linewidth= 1.5)   

ax.plot(X_est, yp_est, 'g-', linewidth= 1.5)



