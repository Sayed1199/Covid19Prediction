# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 08:51:25 2022

@author: Sayed
"""

import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,Activation,MaxPooling2D,Concatenate
from keras.utils import normalize
from keras import Input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


data = np.load('data.npy')
target = np.load('target.npy')


input_shape = data.shape[1:] # 100,100,1
inp = Input(shape = input_shape)
convs=[]
parallel_kernels =[3,5,7]


for k in range(len(parallel_kernels)):
    
    conv = Conv2D(128, parallel_kernels[k], padding='same',activation='relu',input_shape=input_shape,strides=(1,1))(inp)
    convs.append(conv)
  
out = Concatenate()(convs)
conv_model=Model(inputs=inp, outputs=out)

model = Sequential()
model.add(conv_model) 
  
model.add(Conv2D(64,(3,3)))   
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))   
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
  
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,input_dim=128,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint=ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,batch_size=2,callbacks=[checkpoint],validation_split=0.1) 

















