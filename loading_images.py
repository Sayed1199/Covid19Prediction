# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 02:17:22 2022

@author: Sayed
"""

import pandas as pd
import numpy as np
import os
import cv2
                 ##################   preparing datasets   ##################
                 
dataSetPath1='data/covid-chestxray-dataset'       
dataSetPath2='data/covidDataset2'                           
datasetPath='dataset'

categories = os.listdir(datasetPath)

dataset = pd.read_csv(os.path.join(dataSetPath1,'metadata.csv')) 

findings = dataset['finding']
images_names = dataset['filename']

positive_index = np.concatenate((np.where(findings=='Pneumonia/Viral/COVID-19')[0],np.where(findings=='Pneumonia/Viral/SARS')[0]))
positive_images_names = images_names[positive_index]


for index,positive_image_name in enumerate(positive_images_names):
    print(index,'--',positive_image_name)
    image = cv2.imread(os.path.join(dataSetPath1,'images',positive_image_name))
    try:
        cv2.imwrite(os.path.join(datasetPath,categories[1],positive_image_name), image)
    except Exception as e:
        print(e)

     


print('##############################\n#####################\n#################')

    
dataset = pd.read_csv(os.path.join(dataSetPath2,'Chest_xray_Corona_Metadata.csv'))        
findings= dataset['Label']
images_names=dataset['X_ray_image_name']  

negative_index=np.where(findings=='Normal')[0]
negative_images_names = images_names[negative_index]          
    



for index,negative_image_name in enumerate(negative_images_names):
    print(index,'--',negative_image_name)
    image = cv2.imread(os.path.join(dataSetPath2,'Coronahack-Chest-XRay-Dataset','train',negative_image_name))
    try:
        cv2.imwrite(os.path.join(datasetPath,categories[0],negative_image_name),image)
    except Exception as e:
        print(e)
        
        
        
    























