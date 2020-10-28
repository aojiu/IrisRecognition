import cv2
import numpy as np
from IrisLocalization import localization

def normalization(image):
    pupil_x, pupil_y, pupil_radius, iris_radius, draw_img=localization(image)
    draw_img_test = image.copy()
#     iris_center_x, iris_center_y,iris_radius, pupil_center_x, pupil_center_y, pupil_radius = res_lst
    # create the empty normalized image according to the paper
    M=64
    N=512
    normal_img=np.zeros((M,N,3))
    # initialize theta without X
    theta=2*np.pi/N
    # the Y stands for the radius difference
    y_over_m=(iris_radius-pupil_radius)/M
    # compute the area difference
    # reference: https://www.mathopenref.com/coordcirclealgorithm.html
    for Y in range(M):
        for X in range(N):
            # the current angle for points on the circle parameter
            theta_new=theta*X
            # get the points on pupil circle
            # x_p_theta and y_p_theta 
            x_p=np.round(pupil_x+np.cos(theta_new)*pupil_radius,decimals=0)
        
            y_p=np.round(pupil_y+np.sin(theta_new)*pupil_radius,decimals=0)
            # get the points on iris circle
            # x_i_theta and y_i_theta 
            x_i=np.round(pupil_x+np.cos(theta_new)*iris_radius,decimals=0)
            y_i=np.round(pupil_y+np.sin(theta_new)*iris_radius,decimals=0)
            # get x and y in the original image
            orig_x = x_p+(x_i-x_p)*(Y/M)
            orig_y = y_p+(y_i-y_p)*(Y/M)
#             print(orig_y,orig_x)
#             cv2.circle(draw_img_test, (int(orig_x), int(orig_y)), 2, (255, 0, 255), -1)
            # sometimes the result will be slightly larger than boundary
            # eg. 320.25
            if orig_x>=320:
                #print("orig_x exceed limit")
                orig_x=319
            if orig_y>=280:
                #print("orig_y exceed limit")
                orig_y=279
#             print(theta_new,orig_y, orig_x)
#             print(image[int(orig_y)][int(orig_x)])
            
            normal_img[Y][X]=np.array(image[int(orig_y)][int(orig_x)],dtype=int)
    
    normal_img=normal_img.astype("uint8")
#     print(normal_img)
#     return normal_img,draw_img_test
    return normal_img