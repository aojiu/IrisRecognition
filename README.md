# IrisRecognition
Image Analysis Group Project
Weiyao Xie (wx2251), Siyao Zhang (sz2863), Xue Xia (xx2338)

## Getting Started
### Software Requirement
- Python 3.7
- Numpy
- OpenCv 4.4.0
- sklearn 0.22.2
- 


### Usage
Assignment code are stored in different subdirectories. 
```
python assignment1.py
```

For assignment1, input image is stored at ```<image_analysis dir>/assignment1/Homework1```. The output image is stored at ```<image_analysis dir>/assignment1/output```.

## IrisRecognition.py: 
* This is the main function.
* First, read in the images from the relative path and split them into a training group and a testing group.
* For each group, loop through each image. First localize the iris, then normalize and enhance the image. For each enhanced image, extract the features and store the features in the list. Also, store the class of each image in the list.
* Then, do iris matching in different dimensions for three distances to produce Table 3 and Figure 10.
* Calculate false march rate and false non-match rate based on different threshold and plot Table 4 and the ROC.


## IrisLocalization.py: 
* The function localizes the center coordinate of the pupil and iris, and compute the radius for each.
* It first project the original horizontally and vertically to estimate the x,y coordinate of pupil which should be minimum point along each axis. 
* Crop the image into 120*120 centered around estimated center
* Use canny edge to remove noise and use houghcircles to detect exact radius and center coordinates.
* return the center coordinate and radius for both iris and pupil
* Input:Image
* Return:
  * center coordinates and respective radius
  
## IrisNormalization.py
* The function unwrap the iris area into a 64*512 rectangle image
* Create a new empty array to contain the new image
* Compute theta for each coordinate that is between pupil and iris
* get the points on parameter of pupil
```
x_p=np.round(pupil_x+np.cos(theta_new)*pupil_radius,decimals=0)
y_p=np.round(pupil_y+np.sin(theta_new)*pupil_radius,decimals=0)
```
* get the matched points on parameter of iris
```
x_i=np.round(pupil_x+np.cos(theta_new)*iris_radius,decimals=0)
y_i=np.round(pupil_y+np.sin(theta_new)*iris_radius,decimals=0)
```
* map the pixel value from iris area into the new image
```
normal_img[Y][X]=np.array(image[int(orig_y)][int(orig_x)],dtype=int)

```
* Input:
  * Image
  * center coordinates and respective radius
* Return:
  * Normalized image

## IrisEnhancement.py
* divide the normalized image to 32*32 subimage.
* For each block, use histogram equalization to increase the contrast.
* Map all the small blocks back to the original image
* Input:
  * Normalized image
  * Enhanced image
* Return:
  *Normalized image

## FeatureExtraction.py
```
def M1(x ,y, f):
```
* The modulating function of the defined filter 
* Input:
  * x: x coordinates of the pixel
  * y: y coordinates of the pixel
  * f: frequency of the sinusoidal function 
* Return:
  * G: the calculated kernel value
  
```
def Gabor(x, y, f, delta_x, delta_y, i):

```
Gabor filter
* Input:
  * x: x coordinates of the pixel
  * y: y coordinates of the pixel
  * f: frequency of the sinusoidal function 
  * delta_x: space constants of the Gaussian envelope along the x axis 
  * delta_y: space constants of the Gaussian envelope along the y axis 
  * i: identifier of defined filter or Gabor filter 
* Return:
  * G: the calculated kernel value

```
def block(f, delta_x, delta_y, i, size):
```
Calculate the filter grid
* Input:
  * f: frequency of the sinusoidal function 
  * delta_x: space constants of the Gaussian envelope along the x axis 
  * delta_y: space constants of the Gaussian envelope along the y axis 
  * i: identifier of defined filter or Gabor filter 
  * size: grid width of the filter
* Return:
  * w: a size*size grid of filter
```
def features(img):
```
* Read in the enhanced image and make it 48*1536. There are two filters used. The first with delta_x=3 and the second with delta_y=4.5. For both, delta_y=1.5 and f = 1/delta_x. The kernel size is set to 9. Then we extract statistical features in each 8 􏰀*8 small block of the two filtered images and return the features in the list.
* Input
  * img: the enhanced image
* return:
  * f: a list of features, with length equal to 1536
  

## IrisMatching.py: 


```def dim_reduction(feature_train,feature_test,n_comp):```
Reduce the dimension of features to n_comp using LinearDiscriminantAnalysis
* Input:
  * feature_train: a list of features of training class
  * feature_test: a list of features of testing class
  * n_comp: target dimension value
* Return:
  * proj_train: reduced features of  training class with dimension n_comp
  * proj_train: reduced features of  testing class with dimension n_comp

```def adopted_nearest_classifier(feature_train,feature_test,ncomp):```
Nearest center classifier that finds the nearest center in training class to the tested feature
Taking initial rotation angles into consideration. When matching the input feature vector with the templates of an iris class, the minimum of the scores for each rotation angle is taken as the final matching distance. 
* Input:
  * feature_train: a list of features of training class
  * feature_test: a list of features of testing class
  * n_comp: target dimension value, if >=1536, then do not reduce the dimension
* Return:
  * L1_center,L2_center,cosine_center: a list of index of the nearest center for L1, L2 and cosine distance
  * L1, L2, cosine: a list of min distance for L1, L2 and cosine distance


## PerformanceEvaluation.py
```def crr(L1_center,L2_center,cosine_center, labels_train, labels_test):```
Calculate correct recognition rate (CRR) of the matching
* Input:
  * L1_center: a list of index of image  has minimum L1 distance  to each test image
  * L2_center: a list of index of image  has minimum L2 distance  to each test image
  * cosine_center: a list of index of image  has minimum cosine distance  to each test image
  * labels_train: a list of class of train images
  * labels_test: a list of class of test images
* Return:
  * count_L1,count_L2,count_cosine: the CRR for L1, L2 and cosine distance in %

```def roc(threshold, labels_train, labels_test, cosine_center, cosine):```
Calculate the false match rate and false non-match rate of cosine distance to plot ROC curve
* Input:
  * threshold: a list of threshold values
  * labels_train: a list of class of train images
  * labels_test: a list of class of test images
  * cosine_center: a list of index of image  has minimum cosine distance  to each test image
  * cosine: a list of minimum cosine distance of each test image to train images
* Return:
  * FMR: a list of false match rate for each threshold
  * FNMR: a list of false non-match rate for each threshold

## Evaluations
Distance measures:
![distance measures](/evaluations/distance.png)

CRR against Dimensions:
![CRR against Dimensions:](/evaluations/crr.png)

Metrics:
![Metrics:](/evaluations/metrics.png)

ROC:
![ROC:](/evaluations/roc.png)

## Peer Evaluation
![Peer Evaluation:](/evaluations/peer.png)

