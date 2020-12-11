
"""
@author: Sreenivas Bhattiprolu
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
from tqdm import tqdm

import albumentations as A



images_path="GitHub/python_for_microscopists/177/images/" #path to original images
masks_path = "GitHub/python_for_microscopists/177/masks/"
img_augmented_path="GitHub/python_for_microscopists/177/aug_images/" # path to store aumented images
msk_augmented_path="GitHub/python_for_microscopists/177/aug_masks/" # path to store aumented images
images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))

SIZE = 256
aug = A.Compose([
    A.Resize(SIZE, SIZE),
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)

#random.seed(42)

images_to_generate=2000
for i in tqdm(list(range(images_to_generate))): 
    number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
    image = images[number]
    mask = masks[number]
    #print(image, mask)
    #image=random.choice(images) #Randomly select an image name
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

        
    new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
    new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
    io.imsave(new_image_path, transformed_image)
    io.imsave(new_mask_path, transformed_mask)

    
