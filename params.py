# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 5: Vehicle Detection and Tracking
# Date: 13th March 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: params.py
# =========================================================================== #
# Parameters for the classifier, and other global constants

# --------------------------------------------------------------------------- #
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb(98.65 %)
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, None] # Min and max in y to search in slide_window() # no scale-96x96window

# --------------------------------------------------------------------------- #
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'
classifier_filename = 'classifier_pickle.p'