
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
import linear_regression_cls as lir

plt.close('all')


print('plotting data')
df=pd.read_csv('../../data/lreg_data1.txt', names=['population', 'profit'])

data=df.values

X=data[:,0]
y=data[:,1]

fig1, ax1 = plt.subplots()
ltrd=ax1.plot(X,y,'bo', label='Training data')
ax1.set_xlabel('population')
ax1.set_ylabel('profit')

# first_legend = plt.legend(handles=[ltrd], loc='upper right')

print('Program paused. Press enter to continue.\n')
# input()
print('Running Gradient Descent...\n')

m=len(y)
size = 1
for dim in np.shape(X): size *= dim
n=size//m

X.shape = (m, n)
# y.shape=(m,1)

X=np.append(np.ones([m,1]),X, axis=1) # Add a column of ones to x

theta=np.zeros(n+1) # initialize fitting parameters
# Some gradient descent settings
lreg=lir.LReg()
iterations=1500
alpha=0.01
# compute and display initial cost
J=lreg.computeCost(X, y, theta)
print(J)

theta, J_history = lreg.gradientDescent(X, y, theta, alpha, iterations)


print('Theta found by gradient descent: ')

print('{:0.5f},  {:0.5f}'.format(theta[0], theta[1]))

# Plot the linear fit
predicton=X@theta

lregpl=ax1.plot(X[:,1],predicton, '-r', label='Linear regression')
ax1.legend(framealpha=1, frameon=True);

# %% ============= Part 4: Visualizing J(theta_0, theta_1) =============


print('Visualizing J(theta_0, theta_1) ...\n')

#  Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 110);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)]);

# Fill out J_vals
t=np.zeros(n+1)
for i in range(len(theta0_vals)):
    t[0]=theta0_vals[i]
    for j in  range(len(theta1_vals)):
        t[1]=theta1_vals[j]
    
        J_vals[i,j] = lreg.computeCost(X, y, t);
        
T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot_surface(T0, T1, J_vals.T)
  
ax2.set_xlabel('\u03F4_0')
ax2.set_ylabel('\u03F4_1')
ax2.set_zlabel('cost')      



df=pd.read_csv('../../data/lreg_data2.txt', names=['p1', 'p2','p3'])



data=df.values.astype(np.float64)

X=data[:,:-1]
y=data[:,-1]
m=len(y)

size = 1
for dim in np.shape(X): size *= dim
n=size//m

X.shape = (m, n)


X,mu,sigma = lreg.featureNormalize(X)


X=np.append(np.ones([m,1]), X, axis=1)
alpha = 0.005
num_iters = 4000

# Init Theta and Run Gradient Descent 
theta = np.zeros(n+1);
theta, J_history = lreg.gradientDescent(X, y, theta, alpha, num_iters);

fig3, ax3 = plt.subplots()
ltrd=ax3.plot(J_history,'b-', label='Cost History')
ax3.set_xlabel('#Num of Iterations')
ax3.set_ylabel('Cost') 

xtest=np.array([1650., 3.])
xtestn=xtest-mu
xtestn=xtestn/sigma
xtestne=np.insert(xtestn, 0,1)


predict_price=xtestne.T@theta

print('lreg steepest descent preicted price is \n')

print('{:0.2f}'.format(predict_price))


theta_theo=lreg.normalEqn(X, y)

xteste=np.insert(xtest, 0,1)

theo_price=(xtestne.T@theta)

print('lreg theor. preicted price is \n')

print('{:0.2f}'.format(theo_price))





