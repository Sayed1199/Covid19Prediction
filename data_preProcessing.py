# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 08:26:28 2022

@author: Sayed
"""

import os
import cv2
import numpy as np
from keras.utils import np_utils


datasetPath='dataset'
categories = os.listdir(datasetPath)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories,labels))

img_size=100
data=[]
target=[]















for category in categories:
    folder_path = os.path.join(datasetPath,category)
    img_names = os.listdir(folder_path)
    
    for img_name in img_names:
        print(img_name)
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #convert to gray scale
            resized_img = cv2.resize(gray, (img_size,img_size)) #resize img
            data.append(resized_img) # appending image
            target.append(label_dict[category]) # appending labels
            
        except Exception as e:
            print(e)
 
    
 
data = np.array(data)/255.0 # make all pixels between 0 and 1
data = np.reshape(data, (data.shape[0],img_size,img_size,1)) # make it 4d array
target = np.array(target)

new_target = np_utils.to_categorical(target)  # make it categorical

np.save('data',data)
np.save('target',target)

 

 
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
