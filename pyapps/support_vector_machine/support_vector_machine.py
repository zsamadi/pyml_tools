

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:29:231 2021

@author: zayn
"""

## Initialization
import sys
sys.path.append('../../software/algorithms/')

import svm_cls as svm
import scipy.io
import numpy as np
import time
from sklearn.svm import SVC



## =============== Part 1: Loading and Visualizing Data ================
#  We start this example application by first loading and visualizing the dataset. 
#  The following code will load the dataset and plot the data.
#

print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('../../data/svm_data/svmdata1.mat')

X=mat["X"]
y=mat["y"]
y=np.ravel(y)

svmal=svm.svmal()

# Plot training data
ax=svmal.plotData(X, y);

## ==================== Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

y_svm=-(-1)**y

C=10
tol=0.001
max_passes=20
ktype='linear'


svmal=svm.svmal(C=C, ktype=ktype)
svm_sam=svmal.svmz(X, y_svm,tol=0.001, max_passes=20)
alpha_vec=svm_sam.alphas
b=svm_sam.b
alphay=alpha_vec*y_svm
w=alphay@X
svmal.visualizeBoundryLinear(ax, w, b, X)
aa=svmal.score(X, y)
print('train acuracy using developed svm is {0:.2f}%\n'.format(aa*100))



# %% =============== Part 2: Loading and Visualizing Data ================
#  We load another set of data 

print('=============== New Training ================')    

print('Loading and Visualizing Data ...\n')

mat = scipy.io.loadmat('../../data/svm_data/svmdata2.mat')

X=mat["X"]
y=mat["y"]
y=np.ravel(y)
y_svm=-(-1)**y

svmal=svm.svmal()
ax=svmal.plotData(X, y);

# % SVM Parameters

C=1
tol=0.001
max_passes=20
gamma=50
ktype='rbf'


tic=time.time()

svco=SVC(C=C,kernel=ktype, gamma=gamma).fit(X,y)

toc=time.time()
pass_time=toc-tic
print('sklearn svm training lasted {0:.2f}s\n'.format(pass_time))    

        
percentage=svco.score(X,y)
print('train accuracy using sklearn svm is: {0:.2f}% \n'.format(percentage*100))


tic=time.time()

svmal=svm.svmal(C=C, ktype=ktype, gamma=gamma)
svm_sam=svmal.svmz(X, y_svm,tol, max_passes)
b=svm_sam.b
alpha_vec=svm_sam.alphas

toc=time.time()
pass_time=toc-tic
print('svm training lasted {0:.2f}s\n'.format(pass_time))

    

Xe=X[alpha_vec>0]
ye=y_svm[alpha_vec>0]
alpha_vece=alpha_vec[alpha_vec>0]


svmal.visualizeBoundry(ax, alpha_vece, b,Xe,ye, ktype, gamma)

aa=svmal.score(X, y)
print('train acuracy using developed svm is {0:.2f}%\n'.format(aa*100))

# =============== Part 3: Visualizing Dataset 3 ================
# The following code will load the next dataset into and plot the data. 
# 
print('=============== New Training ================')    


print('Loading and Visualizing Data ...')


mat = scipy.io.loadmat('../../data/svm_data/svmdata3.mat')

X=mat["X"]
y=mat["y"]
Xval=mat["Xval"]
yval=mat["yval"]

y=np.ravel(y)
y_svm=-(-1)**y

yval=np.ravel(yval)
ax=svmal.plotData(X, y);

# ========== Training SVM with RBF Kernel  ==========

 
C=1
tol=0.001
max_passes=20
gamma=50
ktype='rbf'

tic=time.time()

svco=SVC(C=C,kernel=ktype, gamma=gamma).fit(X,y)

toc=time.time()
pass_time=toc-tic
print('sklearn svm training lasted {0:.2f}s\n'.format(pass_time))    

        
percentage=svco.score(Xval,yval)
print('validation accuracy using sklearn svm is: {0:.2f}%\n'.format(percentage*100))


tic=time.time()

svmal=svm.svmal(C=C, ktype=ktype, gamma=gamma)
svm_sam=svmal.svmz(X, y_svm,tol, max_passes)
b=svm_sam.b
alpha_vec=svm_sam.alphas


toc=time.time()
pass_time=toc-tic
print('svm training lasted {0:.2f}s\n'.format(pass_time))

    

Xe=X[alpha_vec>0]
ye=y_svm[alpha_vec>0]
alpha_vece=alpha_vec[alpha_vec>0]

svmal.visualizeBoundry(ax, alpha_vece, b,Xe,ye, ktype, gamma)

aa=svmal.score(Xval, yval)
print('validation acuracy using developed svm is {0:.2f}%\n'.format(aa*100))






