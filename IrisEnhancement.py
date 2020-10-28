import cv2
import numpy as np

def enhancement(normal_img):
    # background illumination
    background=normal_img.copy()
    normal_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#     background=np.zeros((normal_gray.shape[0],normal_gray.shape[1]))
#     # set the x, y coord for the first 16*16 box
#     box_x=8
#     box_y=8
#     for i in range((normal_img.shape[1]//8)-1):
# #         if box_x>=64:
# #                 pass
#         box_y=8
#         for j in range((normal_img.shape[0]//8)-1):
#             if box_y>=64:
#                 pass
#             else:
#                 print(box_x,box_y)
#                 mean=np.mean(normal_gray[box_y-8:box_y+8,box_x-8:box_x+8],axis=1)

#                 background[box_y-8:box_y+8,box_x-8:box_x+8]=normal_gray[box_y-8:box_y+8,box_x-8:box_x+8]-mean
#                 box_y+=8
            
            
#         box_x+=8
    
#     new_normal_gray = normal_gray-background
#     new_normal_gray=new_normal_gray.astype("uint8")
    hist_equal=np.zeros((normal_gray.shape[0],normal_gray.shape[1]))
    # set the x, y coord for the first 16*16 box
    box_x=16
    box_y=16
    for i in range((normal_img.shape[1]//8)-1):
#         if box_x>=64:
#                 pass
        box_y=8
        for j in range((normal_img.shape[0]//8)-1):
            if box_y>=64:
                pass
            else:
                
                hist_equal[box_y-16:box_y+16,box_x-16:box_x+16]=cv2.equalizeHist(normal_gray[box_y-16:box_y+16,box_x-16:box_x+16])
                box_y+=8
            
            
        box_x+=8
    
#     new_normal_gray = normal_gray-background
#     new_normal_gray=new_normal_gray.astype("uint8")
    
    
    enhanced_img=cv2.equalizeHist(normal_gray)
    
    
    return hist_equal
            
    
    