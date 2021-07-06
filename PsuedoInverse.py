# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:30:25 2021

@author: Karthik
"""

import numpy as np

#Please make sure that the values of second class are multiplied by -1 
#Since we use sample normalisation for this for example in the below example
# the (-3,1) feature vector belonged to -1 class or class2 hence it is multiplied by -1 and the resulting vector is
# (-1,3,-1)
#SVM1
#c=[74.42,-84.25,-44.86,-47.04,1,84.25,-108.25,-47.5,-50.65,1,44.86,-47.5,-27.88,-29.02,1,47.04,-50.65,-29.02,-30.26,1,1,-1,-1,-1,0]

#SVM2
#c=[-28.01,43.21,1,-43.21,69.7,1,-1,1,0]

#SVM2
c=[0.3247,0,0,1,1,0.0003,0,1,0.8825,0.0022,0,1]

Inp=np.array(c)
Inp=Inp.reshape(3,4)
Y_inv= np.linalg.pinv(Inp)

b=[0.0863,0.2662,0.2362]
Inp1=np.array(b)
Inp1=Inp1.reshape(3,1)

print(np.round(Y_inv,4))
a=np.matmul(Y_inv,b)
print(a)
