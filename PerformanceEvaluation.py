# -*- coding: utf-8 -*-
"""
@author: Xue Xia (xx2338)
"""

import cv2
import numpy as np
import math



def crr(L1_center,L2_center,cosine_center, labels_train, labels_test):
    count_L1 = 0
    count_L2 = 0
    count_cosine = 0
    
    for i in range(0,len(L1_center)):
        L1_index = L1_center[i]
        L2_index = L2_center[i]
        L3_index = cosine_center[i]
        
        # count the number of correct match for each distance
        if (labels_train[L1_index]==labels_test[i]):
            count_L1+=1
        if (labels_train[L2_index]==labels_test[i]):
            count_L2+=1
        if (labels_train[L3_index]==labels_test[i]):
            count_cosine+=1
    # calculate the Correct recognition rate (CCR) in %
    count_L1 = count_L1/len(L1_center)*100
    count_L2 = count_L2/len(L2_center)*100
    count_cosine = count_cosine/len(cosine_center)*100
    
    return count_L1,count_L2,count_cosine




def roc(threshold, labels_train, labels_test, cosine_center, cosine):
    # the list of image  matched correctly
    yes = []
    # the list of image  matched incorrectly
    no = []
    for i in range(0,len(cosine_center)):
        if (labels_train[cosine_center[i]]==labels_test[i]):
            yes.append(cosine[i])
        else:
            no.append(cosine[i])
            
    
    # the list of False match rate
    FMR = []
    # the list of False non-match rate
    FNMR = []
    for t in threshold:
        
        fm = 0
        fnm = 0
        num = 0
        for y in yes:
            if y > t:
                fnm += 1
                num += 1
        for n in no:
            if n < t:
                fm += 1
            else:
                num += 1
        #print(fnm)
        fnmr = fnm*1.0/float(len(yes))
        fmr = fm*1.0/float(len(no))
        #fmr = fm*1.0/float(len(cosine_center)-num)
        #fnmr = fnm*1.0/(float(num))
        
        FMR.append(fmr)
        FNMR.append(fnmr)
        
    return FMR,FNMR

