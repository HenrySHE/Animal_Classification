import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2 as cv
import os
from glob import glob

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
'''
images = np.array(
    [ia.quokka(size=(64, 64)) for _ in range(32)],
    dtype=np.uint8
)
'''

path = '/Users/haoyushe/Code(Local)/LongFuShan_Data/Training/Resized_img/'


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
    data.append(img)



seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

images_aug = seq.augment_images(data)

file_num=0
for file_num in range(len(images_aug)):
    cv.imwrite(os.path.join(path,'Aug_Test'+str(file_num)+'.PNG'),images_aug[file_num])
    file_num+1

#cv.imwrite(os.path.join(path,'Aug_Test.PNG'),images_aug[1])
