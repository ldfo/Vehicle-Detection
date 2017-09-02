### Vehicle detection project
---
### Write-up
The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.

* Run the pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.

* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-not-car.png
[image2]: ./output_images/test4_heatmap.png
[image3]: ./output_images/test4.jpg
[image4]: ./test_images/test4.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


---


###Histogram of Oriented Gradients (HOG)

####1. Feature Extraction
The code is divided into two main files:
1. `vehicle_detection.py` where I run the pipeline for detecting vehicles on a video or an image.
2. `train_classifier.py` where I read the images, extract the features and train a  LinearSVC.

The extraction of features is contained on the functions `extract_features` and `get_hog_features` in lines 25 to 71 of the file called `extract_features.py` which is similar to the examples given in the lesson.

I started by reading in all the `vehicle` and `non-vehicle` images (`extract_features.py` lines 12 to 21).  Here is an example of the `vehicle` and `non-vehicle` classes:

![Vehicle-non-vehicle][image1]

The `YCrCb` colorspace worked well because of the way it splits color channels, it separates color from luminance, which I think makes it good for detecting and features. I chose to use all of the color channels for HOG feature extraction.

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I started with the values used in the lesson and they worked well. This are the final parameters I used:

| Parameter           | Value   |
|---------------------|---------|
| Color Space         | YCrCb   |
| Histogram Bins      | 30      |
| Hog Orientations    | 10       |
| Hog Pixels Per Cell | 8       |
| Hog Cells Per Block | 2       |
| Hog Channel         | All     |
| Spatial size        | 32 x 32 |


####2. Classifier Training

I trained a linear Support Vector Machine. I used LinearSVC from sklearn with default settings of square-hinged loss function and l2 normalization (`train_classifier.py`). The trained model had an accuracy of 98.4%. I exported the trained model, scaler and parameters to a pickle for easy loading when starting the vehicle detection pipeline. That way I don't need to train the model every time.

###Sliding Window Search

On `vehicle_pipeline.py` I defined a class called Vehicle_pipeline, it is a callable
class that takes an image and returns bounding boxes of detected vehicles.

This Vehicle_pipeline works as follows:
1. First, it performs a sliding window search at multiple scales and ROIs using the
feature extractor and the previously trained classifier.

2. Then it draws the areas where the classifier predicted a car onto a heat map.

3. Then it thresholds the heat map to reduce the false-positives.

4. Finally, it returns the bounding boxes of the thresholded heat map.

5. The last step is to draw the boxes on the image, this is done with the draw_boxes
function found at the start of `utils.py`.

Here is a picture of an image before the pipeline
![Original image][image4]

Here is a picture of the stages of the pipeline.
![Pipe stages][image2]

And then the picture with the boxes drawn
![Boxes drawn][image3]
---

### Video Implementation

Here's a [link to my video result](./project_video_processed.mp4), also on [test_images_processed](./test_images_processed) there are the processed images.
The image pipeline and the video pipeline are basically the same but on the video pipeline instead of basing detections on a single frame's heat map, the heat maps
are filtered together using an exponential filter. This filters out jitter and improves the accuracy of the detections.

---

###Discussion

One big drawback of my pipeline is that it is slow. It can only process one frame every 1.5 seconds or so. That is unacceptable for real time detection. I am currently running the program on CPU so maybe running it on GPU or reimplementing some functions can help. Another thing that could help is reducing the window size.

I found difficult to draw the bounding boxes because many of them don't completely enclose the cars. Maybe it's a downside of the thresholding but reducing the heat map threshold introduces some false positives so I decided to leave it like that. With a classifier with fewer false-positives, I could draw better this boxes.

Overall the accuracy and performance of this pipeline look fine, I look forward to using the GPU for speeding up the calculations and maybe process real time video.
