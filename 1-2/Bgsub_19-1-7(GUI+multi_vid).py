# -*- coding: utf-8 -*-
# import the library
from appJar import gui
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

# handle button events
def press(button):
    if button == "Cancel":
        app.stop()
    else:
        path = app.getEntry("Path")
        #pwd = app.getEntry("Password")
        print("path: ", path)
        #path = input("Plz input the video path:")
        print(path)
        #path = ''
        filenames = []
        filenames = [vid for vid in glob(path+"*.MP4")]
        for vid in glob(path+"*.AVI"):
            filenames.append(vid)
        filenames.sort()
        if len(filenames)==0:
            print('No video found in the path!')
            os._exit(0)
        else:
            print('found '+str(len(filenames))+ ' videos in that folder:')

        for video_path in filenames:
            #print(vid_name)
            cv2.ocl.setUseOpenCL(False)
            fileNum = 0


            version = cv2.__version__.split('.')[0]
            #print (version)

            #--------------------------read video file
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            print("length"+str(length))
            print("width"+str(width))
            print("height"+str(height))
            print("fps"+str(fps))

            #------------------Read the basic information of the video
            #t = int(os.path.getctime(video_path))
            #dateArray = datetime.datetime.utcfromtimestamp(t)
            #otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
            #otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
            #print(otherStyleTime)

            #print("Last modified: %s" % time.ctime(os.path.getmtime(video_path)))
            #print("Created: %s" % time.ctime(os.path.getctime(video_path)))
            int_date = int(os.path.getmtime(video_path))
            dateArr = datetime.datetime.utcfromtimestamp(int_date)
            #Date of that video modied
            d = dateArr.strftime("%Y-%m-%d")
            m = dateArr.strftime("%B")
            #Date that the video is 
            t = dateArr.strftime("%H:%M:%S")
            sec = int(math.ceil(length/fps))
            et = dateArr + datetime.timedelta(seconds=sec)
            #print('hello',et.strftime("%H:%M:%S"))



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
            
            print(data.shape)
            print(len(data))
            #------------------------Save problemed video into csv
            if (len(data) == 0):
                write_data = []
                write_data.append(video_path[len(path):])
                write_data.append(m)
                write_data.append(d)
                write_data.append('-')
                write_data.append('-')
                write_data.append(t)
                write_data.append(et.strftime("%H:%M:%S"))
                write_data.append(dateArr.strftime("%H"))
                if (int(dateArr.strftime("%H")) != int(et.strftime("%H"))):
                    write_data.append(et.strftime("%H"))
                else:
                    write_data.append('-')
                write_data.append(str(sec))
                write_data.append('-')
                write_data.append('-')
                write_data.append("* Cannot find animal")
                

                # Write into scv file
                with open('data.csv','a') as csvfile:
                    fieldnames = ['Video_name','Month','Date','Species','Abundance','Starting_Time','Ending_Time','Activity_Hour','Second_Hour','Activity_Time(Sec)','Remarkable_Behavior','Good_Shot?','Remarks']
                    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    #writer.writerow(fieldnames)
                    writer.writerow(write_data)

                continue;

            predictions = model.predict_classes(data)
            print(predictions)

            dic = {'1':'DC','2':'EAP','3':'WP'}
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

            results = [DC, EAP, WP]
            max_val = max(DC,EAP,WP)
            print('---------results------------')
            print(results)
            print('---------max_val------------')
            print(max_val)
            index = 1
            animal_type = 0
            for i in results:
                print(i)
                if  i == max_val:
                    animal_type = index
                    break
                index= index+1


            print('The animal type is :'+ dic[str(animal_type)])
            if(count-BG == 0):
                print ('Unknown Species!')
                prob = -1
            else:
                prob = max_val/(count-BG)
                if (prob < 0.5):
                    print ('The probability is :' + str((max_val/(count-BG))*100)+ '%')
                    print ('Not sure it is type '+ str(animal_type) +', please check it carefully')
                else:
                    print ('The probability is :' + str((max_val/(count-BG))*100)+ '%')




            write_data = []
            write_data.append(video_path[len(path):])
            write_data.append(m)
            write_data.append(d)
            if (count-BG)==0:
                write_data.append('EAP')
            else:
                write_data.append(dic[str(animal_type)])
            write_data.append('1')
            write_data.append(t)
            write_data.append(et.strftime("%H:%M:%S"))
            write_data.append(dateArr.strftime("%H"))
            if (int(dateArr.strftime("%H")) != int(et.strftime("%H"))):
                write_data.append(et.strftime("%H"))
            else:
                write_data.append('-')
            write_data.append(str(sec))
            write_data.append('-')
            if (count-BG)==0:
                    write_data.append('N')
                    write_data.append('*Need checking')
            if (count-BG) != 0:
                if ((max_val/(count-BG))*100 < 50):
                    write_data.append('N')
                    write_data.append('-')
                else:
                    write_data.append('Y')
                    write_data.append('-')


            # Write into scv file
            with open('data.csv','a') as csvfile:
                fieldnames = ['Video_name','Month','Date','Species','Abundance','Starting_Time','Ending_Time','Activity_Hour','Second_Hour','Activity_Time(Sec)','Remarkable_Behavior','Good_Shot?','Remarks']
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                #writer.writerow(fieldnames)
                writer.writerow(write_data)







# create a GUI variable called app
app = gui("Address input form", "400x200")
app.setBg("white")
app.setFont(18)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Welcome,please input the file path")
app.setLabelBg("title", "white")
app.setLabelFg("title", "gray")

app.addLabelEntry("Path")
#app.addLabelSecretEntry("Password")

# link the buttons to the function called press
app.addButtons(["Submit", "Cancel"], press)

app.setFocus("Path")

# start the GUI
app.go()







