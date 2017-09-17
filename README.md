

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_vehicle.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/color_hist.png
[image4]: ./output_images/sliding_windows.png
[image5]: ./output_images/detected_windows.png
[image6]: ./output_images/heatmap.png
[video1]: ./project_video_solution.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

The whole project code is contained in the ipython notebook.
The first half of the notebook covers training the classifier on the udacity image set (~8000 vehicle/non-vehicle images)
The second half of the notebook has code for the video pipeline.

### Training dataset
I used the udacity dataset which contains ~8000 vehicle and ~8000 non-vehicle images. I used the complete dataset.

### Histogram of Oriented Gradients (HOG) and other features
  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of the `vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` based on sample image shown above. The `get_hog_features` function handles this:


![alt text][image2]

I also used features like `color histogram` and `spatial binning`
Here is the visualization of the color historam for the 3 channels (using 32 bins):

![alt text][image3
]


I tried various combinations of parameters so as the get the best accuracy on the test set. My final parameter set was:

`color_space` = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

`orient` = 8  # HOG orientations

`pix_per_cell` = 8 # HOG pixels per cell

`cell_per_block` = 2 # HOG cells per block

`hog_channel` = 0 # Can be 0, 1, 2, or "ALL"

`spatial_size` = (16, 16) # Spatial binning dimensions

`hist_bins` = 32    # Number of histogram bins

`spatial_feat` = True # Spatial features on or off

`hist_feat` = True # Histogram features on or off

`hog_feat` = True # HOG features on or off


#### 3. Training the classifier

I trained a linear SVM using scikits `linerSVC`. I split up the dataset and used 20% as test dataset and remaining for training. My classifier achived an accuracy of `~0.9625`

### Sliding Window Search

For sliding window search, I basically had `three` zones with differing window sizes. The window size in the farthest zone was the smallest. The search ignored all of the top half of the image (y coordinate less than 350). Three sets of windows were created - `windows_far`, `windows_near`, `windows_mid`. These are visualized on the test image below: 

![alt text][image4]

The below image shows the `detections` i got for the above sample image:

![alt text][image5]

After generating the `heatmap` based on the `detections`, the following bounding boxes were visualized:

![alt text][image6]

---

### Video Implementation


Here's a [link to my video result](./project_video_solution.mp4)

![alt text][video1]


#### Filtering

From detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. 

In order to reduce the "jumpiness" of the boxes, I maintained a history of previous heatmaps. While processing every frame, I summed the history and then took a weighted average of the current heatmap and the history

I also added a sanity check to use bounding boxes from previous frame if there are no bounding boxes found in the current frame
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The largest amount of time spent was on tuning the heatmap thresholds and incorporating historic frame information into current frame calculations (with appropriate weights) in order to make the bounding boxes smoother. 

When using YUV color space, My classifer seems to work better on the darker car. This results in some sections of the video not detecting the white car. Part of the problem could be that udacity dataset may have less number of images for white cars. Although my classifier has accuracy of 0.979 , in some sample images, number of `detections` for white car were less compared to the other car.

After changing to `HLS` color space, the white car does register higher number of `detections`. But with `HLS` color space, the classifier also has quite a lot of false positive detections. These were mainly removed by higher level of heatmap thresholding

I also added a sanity step in the pipeline to use bounding boxes found in previous frame if the current frame does not give any detections resulting in bounding boxes.
