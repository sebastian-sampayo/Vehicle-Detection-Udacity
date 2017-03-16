# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 5: Vehicle Detection and Tracking
# Date: 13th March 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: pipeline.py
# =========================================================================== #
# Pipeline for image processing

# Notes:
#    It's not that bad. Major problem: False positives
#    TODO: improvement: scaled windows (taking account perspective).
#   
# Classifier training results:
# ----------------------------
# (C:\Users\Sebastian\Miniconda3\envs\IntroToTensorFlow) C:\Users\Sebastian\Documents\GitHub\Vehicle-Detection-Udacitypython pipeline.py
# Using: 9 orientations 8 pixels per cell and 2 cells per block
# Feature vector length: 6108
# 4.59 Seconds to train SVC...
# Test Accuracy of SVC =  0.9885
# 0.05 Seconds to predict...

# (C:\Users\Sebastian\Miniconda3\envs\IntroToTensorFlow) C:\Users\Sebastian\Documents\GitHub\Vehicle-Detection-Udacitypython pipeline.py
# Using: 9 orientations 8 pixels per cell and 2 cells per block
# Feature vector length: 6108
# 17.21 Seconds to train SVC...
# Test Accuracy of SVC =  0.9854
# 0.05 Seconds to predict...

# --------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.ndimage.measurements import label

from utils import *
from params import *

# --------------------------------------------------------------------------- #
# If True, trains the classifier
TRAIN_MODE = False

# --------------------------------------------------------------------------- #
# Retrieve camera calibration data from previous project (Advanced Lane Lines Finding)
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# --------------------------------------------------------------------------- #
dir_cars = 'data/vehicles'
dir_notcars = 'data/non-vehicles'

test_image = 'test_images/test3.jpg'

if TRAIN_MODE:
    [svc, X_scaler] = train_classifier(dir_cars, dir_notcars)
else:
    # Load classifier from file
    dist_pickle = pickle.load( open(classifier_filename, "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]

image = mpimg.imread(test_image)
# Undistort test image
image = cv2.undistort(image, mtx, dist, None, mtx)

draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255


#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                    xy_window=(96, 96), xy_overlap=(0.7, 0.7))

#windows = slide_scaled_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                    xy_window=(64, 64), xy_overlap=(0.75, 0.75), max_scale=4)

img_shape = image.shape
win_near = np.int(img_shape[0]/4)
xy_window_near = (win_near , win_near)
#xy_window_far = (32, 32)
win_nearfar = np.int(win_near/2)
xy_window_nearfar = (win_nearfar, win_nearfar)
#y_start_stop_far = [350, 450] # far away
y_start_stop_near = [win_near*2, None] # near
y_start_stop_nearfar = [win_near*2, np.int(img_shape[0]*3/4)] # in the middle

windows_near = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_near, 
                    xy_window=xy_window_near, xy_overlap=(0.75, 0.75))

windows_nearfar = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_nearfar, 
                    xy_window=xy_window_nearfar, xy_overlap=(0.75, 0.75))

#windows_far = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_far, 
#                    xy_window=xy_window_far, xy_overlap=(0.75, 0.75))

windows = []
windows.extend(windows_near)
windows.extend(windows_nearfar)
#windows.extend(windows_far)

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

ystart = y_start_stop_near[0]
ystop = img_shape[0] #y_start_stop_near[1]
scale = 1.5
window = win_near
hot_windows_fast_near = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

ystart = y_start_stop_nearfar[0]
ystop = y_start_stop_nearfar[1]
#scale = 1
window = win_nearfar
hot_windows_fast_nearfar = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

hot_windows_fast = []
hot_windows_fast.extend(hot_windows_fast_near)
hot_windows_fast.extend(hot_windows_fast_nearfar)

window_img_debug = draw_boxes(draw_image, windows, color=(0, 200, 0), thick=1, debug=False) 
window_img = draw_boxes(draw_image, hot_windows, color=(0, 200, 0), thick=1, debug=False) 
window_img_fast = draw_boxes(draw_image, hot_windows_fast, color=(0, 200, 0), thick=4, debug=False) 
window_near_debug = draw_boxes(draw_image, windows_near, color=(0, 200, 0), thick=1, debug=False)
window_far_debug = draw_boxes(draw_image, windows_nearfar, color=(0, 200, 0), thick=1, debug=False)

# plt.figure(figsize=(20,10))
# plt.imshow(window_img_fast)
# plt.show()

plt.figure(figsize=(20,10))
plt.imshow(window_img)
plt.savefig('output_images/hot_windows.png')
plt.show()

f = plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(window_near_debug)
plt.title('Sliding windows. Size: {}x{}'.format(win_near, win_near))
# plt.savefig('output_images/windows_near.png')
# plt.show()

# plt.figure(figsize=(20,10))
plt.subplot(122)
plt.imshow(window_far_debug)
plt.title('Sliding windows. Size: {}x{}'.format(win_nearfar, win_nearfar))
# plt.savefig('output_images/windows_nearfar.png')
# plt.show()
f.savefig('output_images/sliding_windows.png')


heat = np.zeros_like(image[:,:,0]).astype(np.float)

# Add heat to each box in box list
heat = add_heat(heat,hot_windows)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
fig.savefig('output_images/heatmap.png')