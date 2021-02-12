
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:35:28 2021

@author: zayn
"""

import sys
sys.path.append('../../software/algorithms/')   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logistic_regression_utils as ut

# importing logistic regression algorithm 
lgreg=ut.lgreg()



#  Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label (Admitted or Not Admitted)

df=pd.read_csv('../../data/logreg_data1.txt', names=['p1','p2','p3'])

data=df.values.astype(np.float64)

X=data[:,:-1]
y=data[:,-1]
m=len(y)

size = 1
for dim in np.shape(X): size *= dim
n=size//m

X.shape = (m, n)



#  Let's start the by first plotting the data to understand the 
#  the problem we are working with.

print('plotting data')

plt.close('all')

ax1=lgreg.plotData(X,y, 'Exam 1 Score', 'Exam 2 Score')

#  implementing the cost and gradient for logistic regression. 


# Setup the data matrix appropriately, and add ones for the intercept term

# y.shape=(m,1)


#  Normalize features
Xn, mu, sig=lgreg.featureNormalize(X)
Xn=np.append(np.ones([m,1]), Xn, axis=1)


#  Initialize fitting parameters
theta=np.zeros(n+1)

#  Compute and display initial cost and gradient
J, grad=lgreg.CostFunction(Xn, y, theta)

print('initial J, cost function is  \n')

print('{:0.5f} \n'.format(J))

#  finding the optimal parameters theta by minimizing  
# the cost function using gradient descent algorithm 

#set parameters for gradient descent algorithm
alpha=0.1
num_iters=400

#   Run gradientDescent to obtain the optimal theta
#   This function will return theta and the cost and the gradient history
theta, J_h=lgreg.gradientDescent(Xn, y, theta, alpha, num_iters)


# Plot the gradient descent history
fig, ax = plt.subplots()
ax.plot(J_h,'b-', label='Cost Function')
ax.set_xlabel('#iteration')
ax.set_ylabel('Cost')   
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

# add computed linear classifier to the data plot
Xe=np.append(np.ones([m,1]), X, axis=1)

ax1.plot(Xe[:,1],-sig[1]*Xn[:,:2]@theta[:2]/theta[2]+mu[1], '-r')

#apply algorithm to predict
#   After learning the parameters, weu'll like to use it to predict the outcomes
#   on unseen data.We will use the logistic regression model
#   to predict the probability that a student with score 45 on exam 1 and 
#   score 85 on exam 2 will be admitted.
# 
#   Furthermore, we will compute the training and test set accuracies of 
#   our model.
# 

#    Predict probability for a student with score 45 on exam 1 
#    and score 85 on exam 2 

exam_score=np.array([45,85])
exam_score.shape=(2,1)

exam_scn=(exam_score-np.mean(exam_score))/sig

exam_scne=np.append([1],exam_scn)
exam_pred=lgreg.sigmoid(theta.T@exam_scne)

print('For a student with scores 45 and 85, we predict an admission ' \
         'probability of \n');
    
print('{:0.2f} \n'.format(exam_pred))


#  Compute accuracy on our training set

exam_pred_train=lgreg.predict(Xn, theta/(theta.T@theta))
num_correct=np.mean(y==exam_pred_train)

print('logistic regression Train Accuracy is : {:0.2f}% \n'.format(num_correct * 100.0));


#  Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

df=pd.read_csv('../../data/logreg_data2.txt', names=['p1','p2','p3'])
data=df.values.astype(np.float64)

X=data[:,:-1]
y=data[:,-1]
m=len(y)

size = 1
for dim in np.shape(X): size *= dim
n=size//m

X.shape = (m, n)

#  Plotting the data to understand the 
#  the problem we are working with.

ax1=lgreg.plotData(X,y, 'Microchip Test 1', 'Microchip Test 2')

#   In this part, we are given a dataset with data points that are not
#   linearly separable. However, we would still like to use logistic 
#   regression to classify the data points. 
# 
#   To do so, we introduce more features to use -- in particular, we add
#   polynomial features to our data matrix (similar to polynomial
#   regression).
# 

#   Add Polynomial Features

#   Note that mapFeature also adds a column of ones for us, so the intercept
#   term is handled

Xn, mu, sig=lgreg.featureNormalize(X)

Xo=lgreg.mapFeature(Xn[:,0], Xn[:,1])

#  Initialize fitting parameters

# y.shape=(m,1)

size = 1
for dim in np.shape(Xo): size *= dim

n1=size//m


#  Set regularization parameter lambda to prevent overfitting

rlambda=1

#  Compute and display initial cost and gradient for regularized logistic
#  regression
theta=np.zeros(n1)

J, grad=lgreg.CostFunction(Xo, y, theta, rlambda)
print('Cost at initial theta (zeros): {:0.2f}\n'.format(J));

#  =============  Regularization and Accuracies =============

# Initialize fitting parameters
rlambda=1
alpha=0.1
num_iters=400
# Optimize
theta, J_h=lgreg.gradientDescent(Xo, y, theta, alpha, num_iters)


# Plot cost history to verify that it's decreasing, and converges
fig, ax = plt.subplots()
ax.plot(J_h,'b-', label='Cost Function')
ax.set_xlabel('#iteration')
ax.set_ylabel('Cost')   
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])


# Plot Boundary

ax=lgreg.plotData(X,y, 'Exam 1 Score', 'Exam 2 Score')


xp0=np.linspace(-1, 1, 100)
xp0.shape=(len(xp0),1)

xp1=np.linspace(-1, 1, 110)
xp1.shape=(len(xp1),1)



Cpr=np.zeros([len(xp0), len(xp1)])

for ii in range(len(xp0)):
    xc0=(xp0[ii]-mu[0])/sig[0]
    for jj in range(len(xp1)):
        xc1=(xp1[jj]-mu[1])/sig[1]
        xcf=lgreg.mapFeature(xc0, xc1)
        Cpr[ii,jj]=lgreg.predict(xcf, theta, True)
        
Xm1, Xm2=np.meshgrid(xp0,xp1)

ax.contour(Xm1, Xm2, Cpr.T, 1)


#  Compute accuracy on our training set

exam_pred_train=lgreg.predict(Xo, theta/(theta.T@theta))
num_correct=np.mean(y==exam_pred_train)

print('Reguralarized linear regression Train Accuracy if : {:0.2f}% \n'.format(num_correct * 100.0));



