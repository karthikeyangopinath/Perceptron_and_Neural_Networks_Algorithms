# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

'''
Update the Inp array
the Inp1 array - exemplars/ data
the Inp2 array - the class label
the Inp3 array - the boundary label
the Inp4 = learning rate
the Inp5 = NUmber of Epochs
the Inp7= Sample normalised for 'SN' or  'NSN' - No sample normalised
'''


import numpy as np

# #The feature Vector Array - always the same if its sample normalised or not - Add zeros before each sample
# Inp=[0,0,2,0,1,2,0,2,1,0,-3,1,0,-2,-1,0,-3,-2]
# xT=np.array(Inp)
# xT=xT.reshape(6,1,3)
# print('\nDummy Class Feature Vector:',xT,'\n')


Inp=[]
no_features=int(input('Enter the number of features: '))
no_instances=int(input('Enter the number of instances: '))
print('Enter the data with enter:')
for i in range(0, no_features*no_instances):
    ele = float(input())
    Inp.append(ele) # adding the element    
print(Inp)


np_zero=np.zeros((no_instances,1))
xT=np.array(Inp)
xT=xT.reshape(no_instances,no_features)
xT=np.hstack((np_zero,xT))
xT=xT.reshape(no_instances,1,no_features+1)
print('\nDummy Class Feature Vector:',xT,'\n')



#The Class Array
Inp1=(1,1,-1,-1)
xT_class=np.array(Inp1)
xT_class=xT_class.reshape(1,no_instances)
xT_class_rows,xT_class_cols=xT_class.shape
print('Class',xT_class,'\n')

#THe initial weight arrays
Inp2=[0.4,1,2]
aT=np.array(Inp2)
aT=aT.reshape(1,3)
print('Initial Weights',aT,'\n')


Inp4=1
learning_rate=Inp4

#Number of  EPOCH
Inp5=2


#Converting the Feature vector Values based on the Class type
def no_sample_norm(xT,xT_class_rows,xT_class_cols):
    for i in range(xT_class_rows*xT_class_cols):
            xT[i][0][0]=1
    return xT



def sample_norm(xT,xT_class_rows,xT_class_cols):
    for i in range(xT_class_rows*xT_class_cols):
        if xT_class[0][i]==1:
            xT[i]=1*xT[i]
            xT[i][0][0]=1
        
        if xT_class[0][i]==-1:
            xT[i]=-1*xT[i]
            xT[i][0][0]=-1
    return xT



#Program ending character
Inp6='S'

#Using SampleNOrmalisation or No Sample Normalisation
Inp7='SN'

if Inp7=='SN':
    xT=sample_norm(xT,xT_class_rows,xT_class_cols)
elif Inp7=='NSN':
    xT=no_sample_norm(xT,xT_class_rows,xT_class_cols)

print("Updated Feature Vector:\n",xT,'\n')

while Inp6 != 't':  
    for i in range(len(Inp1)):
        if Inp7=='NSN':
            if ((np.matmul(xT[i][0].T,aT[0]) >=0) and (xT_class[0][i]<0) ) or (np.matmul(xT[i][0].T,aT[0]) <=0 and xT_class[0][i]>0 ):
                aT_before_update = aT
                aT = aT[0] + learning_rate*xT_class[0][i]*xT[i][0]
                aT=aT.reshape(1,3)
                print('Before Updating',aT_before_update,'  ','After Updating',aT)
            else:
                print('No misclassification')
        
        elif Inp7=='SN':
            #print(xT[i][0].T,'!!!!!!',aT[0],'$$$$$',np.matmul(xT[i][0].T,aT[0]))
            if (np.matmul(xT[i][0].T,aT[0]) <=0):
                aT_before_update = aT
                aT = aT[0] + learning_rate*xT[i][0]
                aT=aT.reshape(1,3)
                print('Before Updating',aT_before_update,'  ','After Updating',aT)
            else:
                print('No misclassification')
    
    Inp6=input('Continue give "y" else "t"')
print('Final weights',aT)