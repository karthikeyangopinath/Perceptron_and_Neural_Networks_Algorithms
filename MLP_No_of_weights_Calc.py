# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

import numpy as np

input_features=2
hidden_layer_nodes=[3]
output_nodes=2

number_of_hidden_layers=len(hidden_layer_nodes)

weight_count=0
weight_count_hidden=0

weight_count=(input_features+1)*hidden_layer_nodes[0]
print('Input and Hidden Layer 1:',weight_count)
for i in range(number_of_hidden_layers):
    if i!=0:
        weight_count_hidden=weight_count_hidden+(hidden_layer_nodes[i]*(hidden_layer_nodes[i-1]+1))

weight_count=weight_count+weight_count_hidden
print('Weights in the hidden layer',weight_count_hidden)    
 
weight_count_output=((hidden_layer_nodes[-1]+1)*output_nodes)
weight_count=weight_count+weight_count_output
print('Weights in the output Layer',weight_count_output)

print('Total Weights',weight_count)  