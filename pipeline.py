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

from utils import *

# --------------------------------------------------------------------------- #
camera_calibration_filename = 'camera_cal/mtx_dist_pickle.p'

# --------------------------------------------------------------------------- #
# Retrieve camera calibration data from previous project (Advanced Lane Lines Finding)
with open(camera_calibration_filename, mode='rb') as f:
    calibration_data = pickle.load(f)
    
mtx, dist = calibration_data['mtx'], calibration_data['dist']

# --------------------------------------------------------------------------- #
