# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:54:23 2018

@author: HenrySHE

"""

# The file name is 'obj_Tracking.py'
import numpy as np
import cv2
import sys
import os
import datetime

'''
def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False
'''

#https://blog.csdn.net/xiaobing_blog/article/details/12591917
video_path = 'RCNX0028.MP4'
t = int(os.path.getctime(video_path))
dateArray = datetime.datetime.utcfromtimestamp(t)
#otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
otherStyleTime = dateArray.strftime("%Y-%m-%d")
print(otherStyleTime)
path = 'C:/Users/user/Desktop/temp'
cv2.ocl.setUseOpenCL(False)
fileNum = 0


version = cv2.__version__.split('.')[0]
print (version)

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
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold	=60)
    #fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

ret,frame = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX

while (cap.isOpened):

    #if ret is true than no error with cap.isOpened
    ret, frame = cap.read()


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
        #-----------------------Combine Countours-----------------------
        LENGTH = len(contours)
        status = np.zeros((LENGTH,1))

        for i,cnt1 in enumerate(contours):
            x = i
            if i != LENGTH-1:
                for j,cnt2 in enumerate(contours[i+1:]):
                    x = x+1
                    dist = find_if_close(cnt1,cnt2)
                    if dist == True:
                        val = min(status[i],status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x]==status[i]:
                            status[x] = i+1
        unified = []
        maximum = int(status.max())+1
        for i in range(maximum):
            pos = np.where(status==i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                unified.append(hull)

        #-----------------------Combine Countours-----------------------
        '''

        # print (contours)
        #looping for contours
        #Doc: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
        #cv2.groupRectangles(rectList, groupThreshold[, eps]) → rectList, weights
        
        #rectList, weights = cv2.groupRectangles(contours, 3, 0.2)
        #temp = np.array(contours).tolist()
        #rectList = cv2.groupRectangles(temp,1,0.2)
        aa = [[1050, 0, 1260, 144], [1085, 0, 1295, 144], [1015, 23, 1225, 168], [1050, 23, 1260, 168], [280, 782, 490, 960]]
        #print (aa)
        #cv2.groupRectangles(aa, 1, 0.7)
        
        for a in contours:
            if cv2.contourArea(a) <2500:
                continue
            #print (a)
            print ('-------------------')
            (x, y, w, h) = cv2.boundingRect(a)
            print ('******'+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'******')
            print ('-------------------')
        
        
        for c in contours:
            if cv2.contourArea(c) >45000:
                continue
            if cv2.contourArea(c) <2500:
                continue
            #get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
            #if(w*h > 500*500):
            #    continue
            #draw bounding box
            if((w*h) > (65*65)):
                crop_img = resized_frame[y:y+h,x:x+w]
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(path,otherStyleTime+'_'+str(fileNum)+'.jpg'),crop_img)
                fileNum = fileNum+1
                # cv2.line(frame,(xx,yy+int(hh/2)),(xx+ww,yy+int(hh/2)),(0,0,255),2)
                # cv2.putText(frame,'Counter:'+str(counter),(0,30),font,0.8,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow('Cropped img',crop_img)
        cv2.imshow('foreground and background',fgmask)
        cv2.imshow('rgb',resized_frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break
        k = cv2.waitKey(30) & 0xff
        if (k == 27):
            break

cap.release()
cv2.destroyAllWindows()
