# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

import numpy as np

input_features=2
hidden_layer_nodes=[3]
output_nodes=2

input_bias=1
hidden_layer_bias=[1]

Inp=[-0.3,-0.5]
x=Inp

#INput weights
Weights_hidden=np.array([1,1,4,3,-3,-1])
Weights_hidden=Weights_hidden.reshape(hidden_layer_nodes[0],1,input_features)
Weight_hidden_bias=np.array([3,-1,-4])
print(Weights_hidden)


#Weight between hiddent and output layer
Weights_last_hidden=np.array([-5,5,-5,1,4,-5])
Weights_last_hidden=Weights_last_hidden.reshape(output_nodes,1,hidden_layer_nodes[-1])
Weight_last_hidden=np.array([2,4])


#Activation Function
y_activation_function = 'LSF'
z_activation_function = 'LTF'

def activation_function(transfer_function,type_transfer):
    if type_transfer=='HLS':
        if transfer_function >=0:
            return 1
        else:
            return 0
    if type_transfer=='LTF':
        return transfer_function
    if type_transfer=='SSF':
        return (2/(1+np.exp(-2*transfer_function)))-1
    if type_transfer=='LSF':
        return (1/(1+np.exp(-transfer_function)))
    if type_transfer=='RBF':
        return np.exp(-(transfer_function*transfer_function))

y=[]

for i in range(hidden_layer_nodes[0]):
    y_transfer_temp=np.sum(x*Weights_hidden[i])+input_bias*Weight_hidden_bias[i]
    y_activation=activation_function(y_transfer_temp,y_activation_function)
    y.append(y_activation)
y=np.array(y)
y=y.reshape(hidden_layer_nodes[0],1,)

print('y',y)
print('---------------')


z=[]
for i in range(output_nodes):
    z_transfer_temp=np.sum(y.T*Weights_last_hidden[i])+(hidden_layer_bias[-1]*Weight_last_hidden[i])
    z_activation=activation_function(z_transfer_temp,z_activation_function)
    z.append(z_activation)

z=np.array(z)
print(z)
    
    
