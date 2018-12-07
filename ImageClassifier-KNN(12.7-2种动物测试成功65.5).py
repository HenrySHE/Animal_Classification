# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:37:49 2018
Updated on Fri Dec 07 16:40:08 2018

@author: HenrySHE

@Aim: Write an KNN program to classifying images

@Reference:
    Plot the result: https://stackoverflow.com/questions/32980675/knn-train-in-cv2-with-opencv-3-0
    https://segmentfault.com/a/1190000007795229
"""

import numpy as np
import cv2 as cv
import sys
import os
import datetime
from glob import glob
from matplotlib import pyplot as plt

#Initialize a array to store the image infomation
data = []
#Read all the files in the folder
filenames = []
#filenames = [img for img in glob("Training_Data/EAP/*.jpg")]
filenames = [img for img in glob("Resized_img/DC_Resized/*.PNG")]
filenames.sort()
#print(filenames)

for img_name in filenames:
    img = cv.imread(img_name)
    #print(img.shape)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #print(gray.shape)
    data.append(gray)

filenames2 = []
#filenames = [img for img in glob("Training_Data/EAP/*.jpg")]
filenames2 = [img for img in glob("Resized_img/WP_Resized/*.PNG")]
filenames2.sort()

for img_name in filenames2:
    img = cv.imread(img_name)
    #print(img.shape)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #print(gray.shape)
    data.append(gray)

'''
filenames3 = []
#filenames = [img for img in glob("Training_Data/EAP/*.jpg")]
filenames3 = [img for img in glob("Resized_img/WP_Resized/*.PNG")]
filenames3.sort()

for img_name in filenames3:
    img = cv.imread(img_name)
    #print(img.shape)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #print(gray.shape)
    data.append(gray)
'''



array_data = np.array(data)
temp = array_data.flatten()
#reshape the image into targeted format
raw_data = temp.reshape(10,100,65,65)
print(raw_data.shape)

#--------------------[Reference Code]------------------------

# Read the Image
#img = cv.imread('digits.png')
# Changed the img from RGB to GREY img
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
#cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
#x = np.array(cells)

#print (x.shape)
#(50,100,20,20)

# Now we prepare train_data and test_data.
#train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
#test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

train = raw_data[:,:50].reshape(-1,4225).astype(np.float32) # Size = (250,2500)
test = raw_data[:,50:100].reshape(-1,4225).astype(np.float32) # Size = (250,2500)

# Create labels for train and test data
k = np.arange(2)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=1)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print(accuracy)

np.savez('knn_EAP_data.npz',train=train, train_labels=train_labels)
# Now load the data
'''
with np.load('knn_EAP_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
    print(train.shape)
    print(train_labels.shape)
'''
