# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

import numpy as np
    
#Backpropagation Algoithm
input_features=2
hidden_layer_nodes=[2]
output_nodes=1

input_bias=1
hidden_layer_bias=[1]

Inp=[0.1,0.9]
x=Inp


#INput weights
Weights_hidden=np.array([0.5,0,0.3,-0.7])
Weights_hidden=Weights_hidden.reshape(hidden_layer_nodes[0],1,input_features)
Weight_hidden_bias=np.array([0.2,0])
print(Weights_hidden)


#Weight between hiddent and output layer
Weights_last_hidden=np.array([0.8,1.6])
Weights_last_hidden=Weights_last_hidden.reshape(output_nodes,1,hidden_layer_nodes[-1])
Weight_last_hidden=np.array([-0.4])


Inp1=0.25
learning_rate=Inp1

Inp2=0.5
truth_value=Inp2

#Activation Function
y_activation_function = 'SSF'
z_activation_function = 'SSF'


#weight_update_hidden_output=-0.4
#weight_update_input_hidden=0.3

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
y_transfer=[]
for i in range(hidden_layer_nodes[0]):
    y_transfer_temp=np.sum(x*Weights_hidden[i])+input_bias*Weight_hidden_bias[i]
    y_transfer.append(y_transfer_temp)
    y_activation=activation_function(y_transfer_temp,y_activation_function)
    y.append(y_activation)
y=np.array(y)
y=y.reshape(hidden_layer_nodes[0],1,)

print('y',y)
print('---------------')


z=[]
z_transfer=[]
for i in range(output_nodes):
    z_transfer_temp=np.sum(y.T*Weights_last_hidden[i])+(hidden_layer_bias[-1]*Weight_last_hidden[i])
    z_transfer.append(z_transfer_temp)
    z_activation=activation_function(z_transfer_temp,z_activation_function)
    z.append(z_activation)

z=np.array(z)
print(z)


def zscore():
    
    y=[]
    y_transfer=[]
    for i in range(hidden_layer_nodes[0]):
        y_transfer_temp=np.sum(x*Weights_hidden[i])+input_bias*Weight_hidden_bias[i]
        y_transfer.append(y_transfer_temp)
        y_activation=activation_function(y_transfer_temp,y_activation_function)
        y.append(y_activation)
    y=np.array(y)
    y=y.reshape(hidden_layer_nodes[0],1,)

    print('y',y)
    print('---------------')


    z=[]
    z_transfer=[]
    for i in range(output_nodes):
        z_transfer_temp=np.sum(y.T*Weights_last_hidden[i])+(hidden_layer_bias[-1]*Weight_last_hidden[i])
        z_transfer.append(z_transfer_temp)
        z_activation=activation_function(z_transfer_temp,z_activation_function)
        z.append(z_activation)

    z=np.array(z)
    return z
    #print(z)


def diff_activation_function(transfer_function,type_transfer):
    if type_transfer=='HLS':
            return 0
    if type_transfer=='LTF':
        return 1
    if type_transfer=='SSF':
        x_temp=-2*transfer_function
        return 4*np.exp(x_temp)/np.power((1+np.exp(x_temp)),2)
    if type_transfer=='LSF':
        return (np.exp(-transfer_function)/((1+np.exp(-transfer_function))*(1+np.exp(-transfer_function))))
    if type_transfer=='RBF':
        return -2*np.exp(-(transfer_function*transfer_function))*transfer_function 


#Paramters to be updated
#Enter the weight between hidden and output layer which needs backprogation
m=Weight_last_hidden

#Enter the details of weights between input to hidden layer whihc needs backprogation
m_to_be_used_for_w =1.6
x_to_be_used_for=x[0]
w=Weights_hidden[1][0][0]


#Program ending character
Inp6='S'

print('Back-propagation Starts:')
while Inp6 != 't': 

    #We need to change the nodes based on the whihc one we are updating
    output_netk = z_transfer[0]
    diff_net=diff_activation_function(output_netk,z_activation_function)
    delta_m =-learning_rate*(z-truth_value)*diff_net
    m=m+delta_m
    print('m',m)



#W weight to be updated
    output_netk = z_transfer[0]
    diff_net_output=diff_activation_function(output_netk,z_activation_function)
    hidden_netj= y_transfer[1]
    diff_net_input=diff_activation_function(hidden_netj,y_activation_function)
    delta_w =-learning_rate*(z-truth_value)*diff_net_output*m_to_be_used_for_w*diff_net_input*x_to_be_used_for
    w=w+delta_w
    print('w',w)
    
    Weight_last_hidden=m
    Weights_hidden[1][0][0]=w
    
    z=zscore()
    print('Updated Score',z)
    Inp6=input('Continue give "y" else "t"')
    

    


