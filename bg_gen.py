# -*- coding: utf-8 -*-
# Background image Generate
# Date: 2018.12.25 Tue
# Author: HenrySHE

import cv2
import os
import datetime
from glob import glob



#https://blog.csdn.net/xiaobing_blog/article/details/12591917
video_path = 'RCNX0061.MP4'
t = int(os.path.getctime(video_path))
dateArray = datetime.datetime.utcfromtimestamp(t)
#otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
print(otherStyleTime)
path = 'temp/'
cv2.ocl.setUseOpenCL(False)

frame_num = 0


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

w1=int(width/5)
h1=int(height/5)
unique_id = 600

ret, frame = cap.read()
while (cap.isOpened):
#if ret is true than no error with cap.isOpened
	ret, frame = cap.read()
	frame_num = frame_num+50
	if ret==True and frame_num >= 10:
		y=0
		while (y<= (height-h1)):
			x=0
			while (x<=(width-w1)):
				crop_img = frame[y:y+h1,x:x+w1]
				cv2.imwrite(os.path.join(path,'Bg_'+otherStyleTime+'_'+str(unique_id)+'.jpg'),crop_img)
				x+=w1
				unique_id+=1
			y+=h1
	if frame_num>=200:
		break
cap.release()
cv2.destroyAllWindows()