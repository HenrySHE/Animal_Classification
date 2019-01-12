# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:29:23 2018
@author: HenrySHE

@Descriptions:
    1. Be able to find the close moving object and capture and save to local folder;
    2. Enlarge the detected bounding box for 50 pixels (larger can achieve more accurate result)

"""

import numpy as np
import csv
import cv2
import math
import sys
import os
import datetime
import os.path, time
import scipy.misc
import datetime
from keras.models import model_from_json
from keras.optimizers import SGD
from glob import glob



#https://blog.csdn.net/xiaobing_blog/article/details/12591917
'''
DC:
video_path = 'RCNX0057.MP4'

[1 1 1 3 1 1 1 1 1 1 1 3 1 3 1 3 2 1 3 2 1 3 3 3 1 1 1 3 3 1 3 3 1 3 1 1 3
 1 1 1 3 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1]
The animal type is :1
The probability is :77.5%

EAP:
video_path = 'RCNX0032.MP4'
[1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
The animal type is :2
The probability is :70.27027027027027%

WP:
video_path = 'IMAG2096.AVI'
video_path = 'IMAG2095.AVI'

[1 1 1 1 3 3 3 3 3 3 3 3 3]
The animal type is :3
The probability is :69.23076923076923%

'''
#video_path = 'RCNX0057.MP4'
#video_path = 'RCNX0032.MP4'
video_path = 'IMAG2096.AVI'
# Human Walks
#video_path = 'IMAG2124.AVI'
#video_path = 'IMAG585.AVI'

#video_path = 'RCNX0031.MP4'


cv2.ocl.setUseOpenCL(False)
fileNum = 0


version = cv2.__version__.split('.')[0]
#print (version)

#read video file
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print("length"+str(length))
print("width"+str(width))
print("height"+str(height))
print("fps"+str(fps))


#check opencv version
if version == '2' :
    fgbg = cv2.BackgroundSubtractorMOG2(varThreshold=60)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
if version == '3':
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold  =30)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
ret,frame = cap.read()
font = cv2.FONT_HERSHEY_SIMPLEX

#-----------------------
#Initialize a empty to store the image data
data = []

#load model
model_architecture = 'structure.json'
model_weights = 'structure_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


#-----------------------

while (cap.isOpened):

    #if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if frame is None:
            break

    if ret==True:

        #Resize the frame to particular size
        resized_frame = cv2.resize(frame,(1000,600))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        #apply background substraction
        fgmask = fgbg.apply(gray)

        #check opencv version
        if version == '2' :
            (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if version == '3' :
            (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        '''
        for a in contours:
            if cv2.contourArea(a) <2500:
                continue
            #print (a)
            print ('-------------------')
            (x, y, w, h) = cv2.boundingRect(a)
            print ('******'+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'******')
            print ('-------------------')
        '''

        for c in contours:
            if cv2.contourArea(c) >35000:
                continue
            if cv2.contourArea(c) <2500:
                continue
            #get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
            #if(w*h > 500*500):
            #    continue
            #draw bounding box
            if((w*h) > (65*65)):
                if(x>50 and y>50):
                    crop_img = resized_frame[y-50:y+h+50,x-50:x+w+50]
                    #cv2.imwrite(os.path.join(path,otherStyleTime+'_'+str(fileNum)+'.jpg'),crop_img)
                    resized_img = cv2.resize(crop_img,(65,65))
                    cv2.rectangle(resized_frame, (x-50, y-50), (x + w +50, y + h+50), (0, 255, 0), 2)
                    fileNum = fileNum+1
                    # cv2.line(frame,(xx,yy+int(hh/2)),(xx+ww,yy+int(hh/2)),(0,0,255),2)
                    # cv2.putText(frame,'Counter:'+str(counter),(0,30),font,0.8,(255,255,255),2,cv2.LINE_AA)
                    #cv2.imshow('Cropped img',crop_img)
                else:
                    crop_img = resized_frame[y:y+h+50,x:x+w+50]
                    #cv2.imwrite(os.path.join(path,otherStyleTime+'_'+str(fileNum)+'.jpg'),crop_img)
                    resized_img = cv2.resize(crop_img,(65,65))
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    fileNum = fileNum+1
                    # cv2.line(frame,(xx,yy+int(hh/2)),(xx+ww,yy+int(hh/2)),(0,0,255),2)
                    # cv2.putText(frame,'Counter:'+str(counter),(0,30),font,0.8,(255,255,255),2,cv2.LINE_AA)
                ###cv2.imshow('resized_img',resized_img)
                #Append the resized image (65*65) into data
                data.append(resized_img)

        ###cv2.imshow('foreground and background',fgmask)
        ###cv2.imshow('rgb',resized_frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

        k = cv2.waitKey(30) & 0xff
        if (k == 27):
            break

cap.release()
cv2.destroyAllWindows()


#re-format the image data
data =np.array(data)/255
#Detection
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
    metrics=['accuracy'])
 
predictions = model.predict_classes(data)
print(predictions)

dic = {'0':'BG','1':'DC','2':'EAP','3':'WP'}
BG=0
DC=0
EAP=0
WP=0
count = 0
for i in predictions:
    if i==0:
        #print('The '+str(count)+' animal is BG')
        BG = BG+1
        
    if i==1:
        #print('The '+str(count)+' animal is DC')
        DC = DC+1
        
    if i==2:
        #print('The '+str(count)+' animal is EAP')
        EAP = EAP +1
        
    if i==3:
        #print('The '+str(count)+' animal is WP')
        WP = WP +1
    count = count +1

results = [BG,DC, EAP, WP]
max_val = max(DC,EAP,WP)
index = 0
animal_type = 0
for i in results:
    if  i == max_val:
        animal_type = index
    break
    index= index+1


print('The animal type is :'+ dic[str(animal_type)])
prob = max_val/(count-BG)
if (prob < 0.5):
    print ('The probability is :' + str((max_val/(count-BG))*100)+ '%')
    print ('Not sure it is type '+ str(animal_type) +', please check it carefully')
else:
    print ('The probability is :' + str((max_val/(count-BG))*100)+ '%')




#t = int(os.path.getctime(video_path))
#dateArray = datetime.datetime.utcfromtimestamp(t)
#otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
#otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
#print(otherStyleTime)

print("Last modified: %s" % time.ctime(os.path.getmtime(video_path)))
print("Created: %s" % time.ctime(os.path.getctime(video_path)))
int_date = int(os.path.getmtime(video_path))
dateArr = datetime.datetime.utcfromtimestamp(int_date)

#--------------------[Success]----------------------
temp = time.localtime(os.path.getmtime(video_path))
print (temp)
print (time.strftime("%Y-%m-%d %H:%M:%S", temp))
#--------------------------------------------------


#Date of that video modied
d = dateArr.strftime("%Y-%m-%d")
m = dateArr.strftime("%B")
#Date that the video is 
#t = dateArr.strftime("%H:%M:%S")
t = time.strftime("%H:%M:%S", temp)
h = time.strftime("%H", temp)
sec = int(math.ceil(length/fps))
#et = dateArr + datetime.timedelta(seconds=sec)
et = time.mktime(temp)+sec
et = time.localtime(et)
et_format = time.strftime("%H:%M:%S",et)
h2 = time.strftime("%H",et)



#print('hello',et.strftime("%H:%M:%S"))


write_data = []
write_data.append(video_path)
write_data.append(m)
write_data.append(d)
write_data.append(dic[str(animal_type)])
write_data.append('1')
write_data.append(t)
write_data.append(et_format)
write_data.append(h)
if (int(h) != int(h2)):
    write_data.append(h2)
else:
    write_data.append('-')
write_data.append(str(sec))
write_data.append('-')
if ((max_val/(count-BG))*100 < 50):
    write_data.append('N')
else:
    write_data.append('Y')


# Write into scv file
with open('data.csv','a') as csvfile:
    fieldnames = ['Video_name','Month','Date','Species','Abundance','Starting_Time','Ending_Time','Activity_Hour','Second_Hour','Activity_Time(Sec)','Remarkable_Behavior','Good_Shot?','Remarks']
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #writer.writerow(fieldnames)
    writer.writerow(write_data)



