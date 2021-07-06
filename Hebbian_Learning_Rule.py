# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:20:00 2021

@author: Karthik
"""

import numpy as np


Inp=[]
no_features=int(input('Enter the number of features: '))
no_instances=int(input('Enter the number of instances: '))
print('Enter the data with enter:')
for i in range(0, no_features*no_instances):
    ele = int(input())
    Inp.append(ele) # adding the element    
print(Inp)


np_zero=np.zeros((no_instances,1))
xT=np.array(Inp)
xT=xT.reshape(no_instances,no_features)
xT=np.insert(xT,0,1,axis=1)
xT=xT.astype(float)
print('\nDummy Class Feature Vector:',xT,'\n')



#THe initial weight arrays
thetha=-1
Inp2=[0,0]
wT=np.array(Inp2)
wT=wT.reshape(1,no_features)
wT=wT.astype(float)
wT=np.insert(wT,0,-thetha,axis=1)
#wT[0][0]=-thetha
print('Initial Weights',wT,'\n')


# The T value or the true value of the parameter
Inp3=[1,1,1,0,0,0]
tT=np.array(Inp3)
tT=tT.reshape(1,no_instances)
tT=tT.astype(float)
tT_rows,tT_cols=tT.shape
print('The T values are :',tT,'\n')

Inp4=1
learning_rate=Inp4


#Number of  EPOCH
Inp5=2  # Dummy Dont consider

#Program ending character
Inp6='S'


y_rounded=[]
wTnew_rounded=[]


print('Before Updating|','weight before update|','y|',' x|','  ','After Updating|','the new weight|')
while Inp6 != 't':   
    for i in range(tT_rows*tT_cols):
        #Heaviside Function calcualtion H(wx) here H(value <=0) =0 else  H(value>0)=1
        y = np.matmul(wT,xT[i].T)
        print(wT,xT[i].T)
        condlist=[y<0,y==0,y>0]
        choicelist=[0,0,1]
        y=np.select(condlist,choicelist)
        y_rounded.append(y.tolist())
        wTnew= wT +  learning_rate*y*xT[i]
        wT_before_update=wT
        wT= wTnew
        print('Before Updating',wT_before_update,y,xT[i],'  ','After Updating',wTnew)
        wTnew_rounded.append(wTnew.tolist())
        
    Inp6=input('Continue give "y" else "t"')

print('Final weights',wT)
#print('y rouned to 4 decimals',np.round(y_rounded,4),'\n')

#print('wT rounded to 4 decimals',np.round(wTnew_rounded,4))