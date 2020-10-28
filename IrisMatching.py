# -*- coding: utf-8 -*-
"""
@author: Siyao Zhang (sz2863), Xue Xia (xx2338)
"""

import numpy as np
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 



def dim_reduction(feature_train,feature_test,n_comp):
    x_train = np.array(feature_train) #vector of 1536

    N_SAMPLES = 108
    y_train = []
    # create classes for training feature vector
    for i in range(0,N_SAMPLES):
        for k in range(0,3):
            y_train.append(i+1)
    y_train = np.array(y_train)

    # fit linear discriminant classifer on training data
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda_object = lda.fit(x_train, y_train) # creates an LDA object for (inputs, targets)

    # transform the training data: Project data to maximize class separation.
    proj_train=lda.transform(feature_train)

    # transform the testing data: Project data to maximize class separation.
    proj_test = lda.transform(feature_test)

    return proj_train,proj_test





def adopted_nearest_classifier(feature_train,feature_test,ncomp):
    # decide whether to reduce the dimension
    if ncomp<1536:
        proj_train,proj_test = dim_reduction(feature_train,feature_test,ncomp)
        f_arr = np.array(proj_train)
        fi_arr = np.array(proj_test)
    else:
        f_arr = np.array(feature_train)
        fi_arr = np.array(feature_test)

    L1_center = []
    L2_center = []
    cosine_center = []
    L1 = []
    L2 = []
    cosine = []
    
    # rotation angles
    #offset = np.arange(-18,19,2)
    offset = np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    #offset = np.array([-9,-6,-3,0,3,6,9])
    
    for i in range(len(fi_arr)):
        L1_dist = []
        L2_dist =  []
        cosine_dist = []
        for j in range(len(f_arr)):

            f = f_arr[j] # train
            fi = fi_arr[i] # test
            
            distoff1 = np.ones(len(offset))
            distoff2 = np.ones(len(offset))
            distoff3 = np.ones(len(offset))
            for k in range(len(offset)):
                distoff1[k] = distance.cityblock(f,np.roll(fi, offset[k]))
                distoff2[k] = distance.euclidean(f,np.roll(fi, offset[k]))
                distoff3[k] = distance.cosine(f,np.roll(fi, offset[k]))
                
            #L1_sum = np.sum([abs(f-fi)])
            L1_dist.append(np.min(distoff1))

            #L2_sum = np.sum((f-fi)**2)
            L2_dist.append(np.min(distoff2))

            #f_norm = np.sum((f)**2)
            #fi_norm = np.sum((fi)**2)
            #cosine_sum = 1 - np.matmul(f,fi)/math.sqrt(f_norm)/math.sqrt(fi_norm)
            cosine_dist.append(np.min(distoff3))
        
        L1_center.append(L1_dist.index(min(L1_dist)))
        L2_center.append(L2_dist.index(min(L2_dist)))
        cosine_center.append(cosine_dist.index(min(cosine_dist)))
        L1.append(min(L1_dist))
        L2.append(min(L2_dist))
        cosine.append(min(cosine_dist))
    
    # return the index and min distance
    return L1_center,L2_center,cosine_center, L1, L2, cosine