# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

import numpy as np

Inp=[]
no_features=int(input('Enter the number of features: '))
no_instances=int(input('Enter the number of instances: '))
no_output_units=int(input('Enter the number of output units: '))
print('Enter the data with enter:')
for i in range(0, no_features*no_instances):
    ele = int(input())
    Inp.append(ele) # adding the element    
print(Inp)



xT=np.array(Inp)
xT=xT.reshape(no_instances,no_features)
xT=xT.astype(float)
xT_rows,xT_cols=xT.shape
print('\nDummy Class Feature Vector:',xT,'\n')



#THe initial weight arrays
#thetha=-1
Inp2=[1,1,0,1,1,1]
wT=np.array(Inp2)
wT=wT.reshape(no_output_units,no_features)
wT=wT.astype(float)
#wT=np.insert(wT,0,-thetha,axis=1)
#wT[0][0]=-thetha
print('Initial Weights',wT,'\n')

'''Not needed for the exam
# # The T value or the true value of the parameter
# Inp3=[1,1,1,0,0,0]
# tT=np.array(Inp3)
# tT=tT.reshape(1,no_instances)
# tT=tT.astype(float)
# tT_rows,tT_cols=tT.shape
# print('The T values are :',tT,'\n')


Inp4=1
learning_rate=Inp4


#Number of  EPOCH
Inp5=2  # Dummy Dont consider
'''


#Program ending character
Inp6='S'

#Alpha parameter
Inp7=0.25
alpha=Inp7

#Beta parameter
Inp8=1
beta=Inp8

y_rounded=[]
wTnew_rounded=[]
y=np.zeros((no_output_units,1))


while Inp6 != 't':
    for i in range(xT_rows):
        
        e=xT[i] - np.matmul(wT.T,y).T
        print('e:',e)
        y= y+alpha*np.matmul(wT,e.T)
        print('We',np.matmul(wT,e.T))
        #print('y',y)
        y_rounded.append(y.tolist())
        print('y',y)
    
        
    Inp6=input('Continue give "y" else "t"')
    
wTnew= wT +  beta*np.matmul(y,e)
wT_before_update=wT
wT= wTnew
print('Before Updating',wT_before_update,'  ','After Updating',wTnew)
wTnew_rounded.append(wTnew.tolist())

# print('Final weights',wT)
#print('y rouned to 4 decimals',np.round(y_rounded,4),'\n')

#print('wT rounded to 4 decimals',np.round(wTnew_rounded,4))