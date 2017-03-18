TODO:
- lines in the code

# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_color_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/hot_windows.png
[image5_example]: ./examples/bboxes_and_heat.png
[image5]: ./output_images/heatmap.png
[image6]: ./examples/labels_map.png
[image7_example]: ./examples/output_bboxes.png
[image7]: ./output_image/final_box.png
[frames]: ./output_image/frames.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


####2. Explain how you settled on your final choice of HOG parameters.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I tried various combinations of parameters and found out that the best case for classification in this application was using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. I also used all channels for the hog features extraction and spatial resize of 32x32 pixels. In the next figure we can see the result of the hog feature extraction in that color space, for each channel of a couple of example images:

![alt text][image2]

Other color spaces that worked well too were HLS, HSV, YUV and LUV. Using those parameters and all the color channels for HOG extraction the resulting feature vector length is 8460.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC()` with the default parameters (C=1 for example). For the training and test data I used a combination of 
[the GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), 
[the KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), 
and examples extracted from the project video itself.
In particular, for the test set, I extracted the 20% of the total number of images in the dataset, using `train_test_split` from the `sklearn.model_selection` module.
With the suggested parameters, the accuracy of the validation after training was about 98.87%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used two sizes of windows to search for cars in the image from the camera. A middle one of 90x90 to search for far and near-far vehicles, and a big one of 180x180 for the nearest vehicles. Furthermore, I set an overlap of 80% of the window, so I can detect smoother changes when a vehicle moves around the image.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

In order to improve the performance of the algorithm, I tried using HOG subsampling. This way, instead of extracting HOG features for each window, I extract HOG features for the whole image and then sub sample that to obtain the windows features. Then, scaling the original image turns into scales of the window we want to slide. The trickiest part is to tune the overlapping. In this case, we can modify the overlapping by changing the number of cells to skip when sliding the window for each step of the algorithm.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a 
[YouTube link to my video result](https://youtu.be/zxqPGv7t-no)
and 
[here](./output_project_video.mp4) 
is a link to the .mp4 file. Both were produced using the original algorithm (not the HOG sub sampling one).


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Then, I wrote an algorithm to save the hot windows found in 12 consecutive frames of the video to implement a kind of "moving average". I created a heatmap with all this windows and increased the threshold to identify vehicle positions. For each new frame processed, I took out the hot windows of the older frame and append the new hot windows, in a LIFO buffer fashion. I tried several sizes for the buffer looking for a tradeof between the clean up of false positives and a quick response to vehicle movement. If the buffer is too large, less false positives appear, but there is a little lag for the final box when the cars change their position quickly. I also considered that the video has approximately 25 frames per second, so taking 12 frames for the buffer is like taking the average over the last half second.

[//]: # (Here are six frames and their corresponding heatmaps:)
[//]: # (Here's an example result showing the heatmap from a frame of the video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid:)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][frames]

[//]: # (### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:)
[//]: # (![alt text][image6])

[//]: # (In the next image the resulting bounding boxes are drawn onto that same frame of the video:)

[//]: # (![alt text][image7])


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

[//]: # (Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  )
It would be cool to work long time with the filtering alogirthm processing a buffer of frames in the video.
Maybe we could get the output probabilities of the classifier (instead of a binary output) to fill in a continuous heatmap. This way, the final tracking box would be moving softer.
[//]: # (We could for example, track the position of the center of the blob instead of the maximum and minimum of the thresholded heatmap, in order to make softer the movement of the final box surrounding the car. )
[//]: # (It will also be interesting to investigate deeper the classification of white cars, that appears to be a little more difficult than other cars. )
However, the classifier seems to achieve its objective identifying cars on the road. This result is amazing and really motivating to move forward on this subject.


