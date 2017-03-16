# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 5: Vehicle Detection and Tracking
# Date: 13th March 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: main.py
# =========================================================================== #
# Main file
'''
'''

# --------------------------------------------------------------------------- #
from moviepy.editor import VideoFileClip
import pickle
import numpy as np

from utils import *
from params import *

# --------------------------------------------------------------------------- #
input_video_filename = 'project_video.mp4'
output_video_filename = 'output_' + input_video_filename

# --------------------------------------------------------------------------- #
# Retrieve camera calibration data from previous project (Advanced Lane Lines Finding)
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# Load classifier from file
dist_pickle = pickle.load( open(classifier_filename, "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]

# A list to accumulate hot windows
hot_windows_aux = []
N_heat = 50
for i in range(N_heat):
    # Init with empty lists.
    hot_windows_aux.append([])
  
# This is the function that will process each image in the video
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below

    # Pipeline!
#    new_hot_windows = pipeline(image, mtx, dist, svc, X_scaler)
    new_hot_windows = pipeline_fast(image, mtx, dist, svc, X_scaler)
    
    # Accumulate the hot windows to 
    hot_windows_aux.pop()
    hot_windows_aux.insert(0, new_hot_windows)
    # Extend list of lists to a simple list
    hot_windows = []
    for i in range(N_heat):
        hot_windows.extend(hot_windows_aux[i])
    
    # Heat map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, np.int(N_heat*1.5))
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_image = np.copy(image)
    result = draw_labeled_bboxes(draw_image, labels, color=(0, 200, 0), thick=5)
#    result = np.dstack((heatmap, heatmap*0, heatmap*0))

    return result

# Load video and process every image.
clip1 = VideoFileClip(input_video_filename)#.subclip(20,40)
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output_video_filename, audio=False)