# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 5: Vehicle Detection and Tracking
# Date: 13th March 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: analysis.py
# =========================================================================== #
# Analysis of parameters for the classifier

# --------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.ndimage.measurements import label

from utils import *
from params import *

# --------------------------------------------------------------------------- #
# Retrieve camera calibration data from previous project (Advanced Lane Lines Finding)
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']


# --------------------------------------------------------------------------- #
# Read in cars and notcars
car = 'test_images/car.png'
notcar = 'test_images/notcar.png'
car_img = mpimg.imread(car)
notcar_img = mpimg.imread(notcar)

# 1st figure. Car - Not car
f, axarr = plt.subplots(1, 2)
axarr[0].set_title('Car', fontsize=20)
axarr[0].imshow(car_img)
axarr[1].set_title('Not car', fontsize=20)
axarr[1].imshow(notcar_img)
plt.show()
# f.savefig('output_images/car_not_car.png')

# Color hist example
car_color_img = convert_RGB2X(car_img, color_space=color_space)
notcar_color_img = convert_RGB2X(notcar_img, color_space=color_space)

# HOG example
f, axarr = plt.subplots(2, 4)
axarr[0,0].set_title('Car', fontsize=20)
axarr[0,0].imshow(car_img)
axarr[1,0].set_title('Not car', fontsize=20)
axarr[1,0].imshow(notcar_img)
for i in range(3):
    car_ch = car_color_img[:,:,i]
    notcar_ch = notcar_color_img[:,:,i]
    
    car_features, car_hog_image = get_hog_features(car_ch, orient, pix_per_cell, cell_per_block, 
                                        vis=True, feature_vec=True)
    notcar_features, notcar_hog_image = get_hog_features(notcar_ch, orient, pix_per_cell, cell_per_block, 
                                        vis=True, feature_vec=True)
    
    axarr[0,i+1].set_title('HOG image - Channel {}'.format(i), fontsize=20)
    axarr[0,i+1].imshow(car_hog_image)
    axarr[1,i+1].set_title('HOG image - Channel {}'.format(i), fontsize=20)
    axarr[1,i+1].imshow(notcar_hog_image)
    
plt.show()
# f.savefig('output_images/HOG_color_example.png')