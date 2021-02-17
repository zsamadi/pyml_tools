
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:35:28 2021

@author: zayn
"""

import sys
sys.path.append('../../software/algorithms/') 
sys.path.append('../../software/utilities/')     

import neural_network_class as snetc
import mnist_data_loader
import matplotlib.pyplot as plt 




training_data, validation_data, test_data = \
    mnist_data_loader.load_data()


training_data=list(training_data)

    


    
snet=snetc.nnet([784, 30,10])


ax=snet.plotData(training_data)



iweights=snet.weights
ibiases=snet.biases

                


rlambda=5.0

alpha=0.5
num_passes=30
batch_size=10
# test_data=None

snet.nnGraDescent(training_data,num_passes, 
                  batch_size,alpha,test_data, rlambda)

fig, ax = plt.subplots()
ax.plot(range(len(snet.J_h)), snet.J_h)




# cr_prec=sum(predictnno.reshape(y.shape)==y)/len(y)*100
# print('\nTraining Set Accuracy: {:0.0f}\n',cr_prec[0] )



