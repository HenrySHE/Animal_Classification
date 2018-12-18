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

@To-Do List    
  1. 应该需要准备多一个识别背景的category
  2. 做一个让用户选择画面的程序，然后自动识别写入（找如何让用户随意选择一个frame)
    2.1. 写入视频的时候，每个视频有一个独立的ID，需要检查是否已经存在，存在的话，同一物种+1，否则减一
