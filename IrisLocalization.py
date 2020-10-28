import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import glob
import math
import scipy
from scipy.spatial import distance
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from tabulate import tabulate

def localization(image):
    # convert the image to grayscale
    boundary=[]
    centers=[]
    draw_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = draw_img.copy()

    center_x=np.argmin(np.mean(img,0))
    center_y=np.argmin(np.mean(img,1))

    #recalculate of pupil by concentrating on a 120X120 area
    center_crop_x = img[center_x-60:center_x+60]
    center_crop_y = img[center_y-60:center_y+60]
    
    crop_center_x=np.argmin(np.mean(center_crop_y,0))
    crop_center_y=np.argmin(np.mean(center_crop_x,0))
    
    # we have the estimated pupil center coord
    est_pupil_center = (crop_center_x,crop_center_y)

    # canny edge to remove noise
    _,bi_image_test = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(bi_image_test, 100, 220)

    # get the circle for pupil
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 10, 100)



    min_distance=2**31
   
    for circle_coord in circles[0]:
        # get the closest circle to the approximate center
        hough_circle_center=(circle_coord[0],circle_coord[1])
        center_distance = distance.euclidean(est_pupil_center, hough_circle_center)
        if center_distance<min_distance:
            min_distance=center_distance
            circle_num=circle_coord

    #draw the inner boundary
    pupil_x, pupil_y, pupil_radius = int(circle_num[0]), int(circle_num[1]), int(circle_num[2])
    cv2.circle(draw_img, (int(circle_num[0]), int(circle_num[1])), int(circle_num[2]), (255, 0, 0), 3)


    #draw the outer boundary, which is approximately found to be at a distance 53 from the inner boundary 
    iris_radius=pupil_radius+53
    cv2.circle(draw_img, (int(circle_num[0]), int(circle_num[1])), pupil_radius+53, (255, 0, 0), 3)

    return pupil_x, pupil_y, pupil_radius, iris_radius, draw_img