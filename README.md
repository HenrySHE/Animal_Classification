# Animal Classification

> created at 2018/12/7 Fri
>
> @Author: Henry SHE
>
> @Description: An project that used to do animal classification (3-4 types of wild animals, using IR cameras). The new features and project log will be listed in this file.

## To-Do List
- Animal Classfication
- Background Subtraction solution(the given algirithm of BS is not good enough)

## Project Log

**2018/12/06**
- Using HOG and SVM to do animal recognition, but didn't have very good performance


**2018/12/07**
- Finishing using KNN to classify 2 types of animals WP and DC (but only have 65% of accuracy)
Reasoning:
- The training data has too much noise

**2018/12/15**
- Finish do the data augmentation (see file img_aug_success.py)
- Build a simple neural network for training (but the validation accuracy is changing a lot during the training) (see the Training.ipynb)

**2018/12/16**
- Uploading the `Training.ipynb` and `Testing_animal.ipynb`, and the weight and the model structure are also saved in this repository. Just edit the image path of the `Testing_animal.ipynb` can get the result. And it is able to classify 3 types of animal, DC,WP and EAP.
- I forget to add the image resize function in the `Testing_animal.ipynb` file, and the test image format should be (65,65,3) format.

Problems:
- The data of DC is not clear enough, so there is some problem when judging DC. The rest of two can reach 95% accuracy or above.
- Still need to add the sliding window to the program.
- Add function allow passing paremeters when calling the `.py` file.

**2018/12/17**
- Upload `Bgsub_18-12-17.py` and it is a background subtraction file, used to do background subtraction, still have some problems. I added a threshold when doing background subtraction, and it shows that when the threshold=60 the performance is the best, when the countour is smaller than 25000, it will be ignored, and still need to find a good way to combine the contours for object dections.

**2018/12/18**
- Upload `nms.py`, means `non maximal suppression` a method to combine the similar bounding boxing technique (worked)
- Upload `Bgsub_18-12-18.py` and it has the following updates:
    1. Be able to find the close moving object and capture and save to local folder;
    2. Enlarge the detected bounding box for 50 pixels (larger can achieve more accurate result)


**2018/12/25**
1. Write `bg_gen.py` for generating the background categories


**2019/1/2** (Happy New Year)
1. Upload folder `1-2` and combine the detection and the background subtraction part, be able to detect animal, for those probability lower then 50% will marked as a unknown, and telling user to judge it manully.
2. Prepare the insallation requirements : `requirements.txt`

To-Do list:
- Letting user to define the video path (or folder).
- Writting the data as csv file part need to be implemented.


**2019/1/4**
1. Add `Bgsub_19-1-4(multiple_vid).py` and `Bgsub_19-1-4(single_vid).py` both of them are able to read video (including in `MP4` format and `AVI` format. 
2. User need to specify the video path before executing the animal detection. and need to add `/` at the end of the file path (e.g. `path/to/your/videos/`) otherwise the program will not read any videos and terminate itself.
3. Still got some problem when detection the background.
4. The accuracy need to be improved.

**2019/1/14**
1. Update the 1-14 version code, correct the timing problem (the older version have timing problem), and I changed the method of using time method, now it can show the correct time information.
2. Upload the `h5` file, it add another categories called `other` the training data is including `MPC` and also `FB` if the over 70% of recognition belongs to `other`, then it might belongs to `other` categories.
3. Upload `requirement.txt` in folder `1-21`, to install the package please type `pip install -r requirement.txt` and it will intall the related packages automatically. (*pay attention that you need to go to the folder of the requirement.txt and execute the command*)

