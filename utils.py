# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 5: Vehicle Detection and Tracking
# Date: 13th March 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: utils.py
# =========================================================================== #
# Utility functions

import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import time

from params import *

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# --------------------------------------------------------------------------- #
def convert_RGB2X(image, color_space='HSV'):
    if color_space == 'HSV':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'BGR':
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
#        print(':convert_RGB2X(): Color space ', color_space, ' unkown. Copying image.')
        output = np.copy(image)
    return output

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                          vis=False, feature_vec=True):
    """
    Define a function to return HOG features and visualization
    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def bin_spatial(img, size=(32, 32)):
    """
    Define a function to compute binned color features  
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Define a function to compute color histogram features 
    NEED TO CHANGE bins_range if reading .png files with mpimg!
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    """
    Define a single function that can extract features using hog sub-sampling and make predictions
    This is a more efficient method for doing the sliding window approach, 
    one that allows us to only have to extract the Hog features once. 
    The code below defines a single function find_cars that's able to both extract features and make predictions.
    """
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
#    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_RGB2X(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
#                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                # Return windows to process or draw later (instead of drawing directly)
                box = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart))
                on_windows.append(box)
                
    return on_windows

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
# I have made some modifications
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Define a function to extract features from a list of images
    Have this function call bin_spatial() and color_hist()
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            feature_image = convert_RGB2X(image, color_space)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Define a function that takes an image,
    start and stop positions in both x and y, 
    window size (x and y dimensions),  
    and overlap fraction (for both x and y)
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# --------------------------------------------------------------------------- #
def slide_scaled_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5), max_scale=3):
    """
    Define a function that takes an image,
    start and stop positions in both x and y, 
    window size (x and y dimensions),  
    and overlap fraction (for both x and y)
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    x_start = x_start_stop[0]
    x_stop = x_start_stop[1]
    y_start = y_start_stop[0]
    y_stop = y_start_stop[1]
    xspan = x_stop - x_start
    yspan = y_stop - y_start
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    xy_window0 = xy_window
    starty = y_start
    print('nywindows: ', ny_windows)
    print('y_window: ', xy_window[1])
    print('max y_window: ', xy_window0[1]*(max_scale))
    print('yspan: ', yspan)
    print(yspan-xy_window0[1]*(max_scale))
    counter = 0
#    est = ny_windows * xy_overlap[1] / max_scale
#    print('estimation: ', est)
    for ys in range(ny_windows):
        # Compute the number of pixels per step in y with the previous window size
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
#        print(ny_pix_per_step)

#        print('starty: ', starty)
#        print('starty-y_start: ', starty-y_start)
        # Scale depending on y-axis position. Lower y: 1 scale. Higher y: max_scale (approximate)
        scale = (1 + (starty-y_start)/(yspan-xy_window0[1]*(max_scale*1.1))*(max_scale - 1))
        # Resize window
        xy_window = (xy_window0[0] * scale, xy_window0[1] * scale)
#        print(xy_window)
        
        # Compute the number of pixels per step in x with the new window size
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        
        # Calculate window y position
        starty = np.int(starty + ny_pix_per_step)
        endy = starty + np.int(xy_window[1])

        # If windows start out of the image, break the loop
        if (starty > y_stop):
            break # early break

        # If the window gets out of the image, don't add it
        if (endy > y_stop):
            continue

        print('actual ywin size: ', endy-starty)
        counter += 1
        
        for xs in range(nx_windows):
            # Calculate window x position
            startx = np.int(xs*nx_pix_per_step) + x_start
            endx = startx + np.int(xy_window[0])
            # If windows start out of the image, break the loop
            if (startx > x_stop):
                break # early break

            # If the window gets out of the image, don't add it
            if (endx > x_stop):
                continue
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    print('final counter: ', counter)
    # Return the list of windows
    return window_list

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6, debug=False):
    """
    Define a function to draw bounding boxes
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    
    if debug:
        color = (250, 0, 0)
#        color = 200
#        print(color)
#        plt.figure(figsize=(20,10))
#        plt.imshow(imcopy)
#        plt.show()

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        
        if debug:
            plt.figure(figsize=(20,10))
            plt.imshow(imcopy)
            plt.show()
#            time.sleep(0.1)
            color = (250, bbox[0][1] * 255/800, 0)
    # Return the image copy with boxes drawn
    return imcopy

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    """
    Define a function to extract features from a single image window
    This function is very similar to extract_features()
    just for a single image rather than list of images
    """
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = convert_RGB2X(img, color_space)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    """
    Define a function you will pass an image 
    and the list of windows to be searched (output of slide_windows())
    """
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# --------------------------------------------------------------------------- #
def predict_single_img(img, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Predict whether the image corresponds to a vehicle or not
    :img: Input image to classify
    :clf: Classifier
    :prediction: Classifier output
    """
    #4) Extract features for that window using single_img_features()
    features = single_img_features(test_img, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    #5) Scale extracted features to be fed to classifier
    test_features = scaler.transform(np.array(features).reshape(1, -1))
    #6) Predict using your classifier
    prediction = clf.predict(test_features)
    return prediction

# --------------------------------------------------------------------------- #
# train function - save pickle with classifier.
def train_classifier(dir_cars='.', dir_notcars='.'):
    """
    Trains a classifier with images 
    :dir_cars: Path to the directory with car images
    :dir_notcars: Path to the directory with non-car images
    :svc: Linear Support Vector Machine Classifier trained
    :X_scaler: A per-column fitted scaler using StandardScaler()
    """
    # Read in cars and notcars
    cars = []
    notcars = []

    print('Reading files...')
    for filename in glob.iglob(dir_cars + '/**/*.png', recursive=True):
            cars.append(filename)
    
    for filename in glob.iglob(dir_notcars + '/**/*.png', recursive=True):
            notcars.append(filename)

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    # sample_size = 50
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    print('Extracting features...')
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets (80%-20% respectively)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    print('Training...')
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('Done. ', round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)*100, '%')
    # Check the prediction time for a single sample
    t=time.time()
    print(round(t-t2, 2), 'Seconds to predict...')
    
    # Save the classifier and scaler for later use
    dist_pickle = {}
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    pickle.dump( dist_pickle, open(classifier_filename , "wb" ) )


    return [svc, X_scaler]

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
    
# --------------------------------------------------------------------------- #
# This function was copied from the Udacity Nanodegree course.
def draw_labeled_bboxes(img, labels, color=(0, 250, 0), thick=6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img

# --------------------------------------------------------------------------- #
# Pipeline for vehicle detection
# Expects an RGB input image
def pipeline(image, mtx, dist, clf, scaler):
    # Undistort test image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    # Define windows size proportional to image shape
    img_shape = image.shape
    win_near = np.int(img_shape[0]/4)
    xy_window_near = (win_near , win_near)
    win_nearfar = np.int(win_near/2)
    xy_window_nearfar = (win_nearfar, win_nearfar)
    y_start_stop_near = [win_near*2, None] # near
    y_start_stop_nearfar = [win_near*2, np.int(img_shape[0]*3/4)] # in the middle
    
    windows_near = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_near, 
                        xy_window=xy_window_near, xy_overlap=(0.75, 0.75))
    
    windows_nearfar = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop_nearfar, 
                        xy_window=xy_window_nearfar, xy_overlap=(0.75, 0.75))
    
    windows = []
    windows.extend(windows_near)
    windows.extend(windows_nearfar)

    hot_windows = search_windows(image, windows, clf, scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

#    result = draw_boxes(draw_image, hot_windows, color=(0, 200, 0), thick=1, debug=False)

    return hot_windows
    
# --------------------------------------------------------------------------- #
# Pipeline for vehicle detection
# Expects an RGB input image
def pipeline_fast(image, mtx, dist, clf, scaler):
    # Undistort test image
    image = cv2.undistort(image, mtx, dist, None, mtx)
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
#    image = image.astype(np.float32)/255

    # Define windows size proportional to image shape
    img_shape = image.shape
    win_near = np.int(img_shape[0]/4)
    y_start_stop_near = [win_near*2, None] # near
    y_start_stop_nearfar = [win_near*2, np.int(img_shape[0]*3/4)] # in the middle

    ystart = y_start_stop_near[0]
    ystop = img_shape[0] #y_start_stop_near[1]
#    scale = 3 # => sliding window size: 192
    scale = 2.25 # => sliding window size: 144
#    new_hot_windows_near = find_cars(image, ystart, ystop, scale, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
    
    ystart = y_start_stop_nearfar[0]
    ystop = y_start_stop_nearfar[1]
    scale = 1.5 # => sliding window size: 96
    new_hot_windows_nearfar = find_cars(image, ystart, ystop, scale, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
    
#    ystart = y_start_stop_nearfar[0]
#    ystop = y_start_stop_nearfar[1]
#    scale = 0.5
#    new_hot_windows_far = find_cars(image, ystart, ystop, scale, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
    new_hot_windows = []
#    new_hot_windows.extend(new_hot_windows_near)
    new_hot_windows.extend(new_hot_windows_nearfar)
#    new_hot_windows.extend(new_hot_windows_far)
    
    return new_hot_windows
