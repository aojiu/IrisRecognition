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
from IrisEnhancement import enhancement
from IrisLocalization import localization
from IrisNormalization import normalization
from IrisMatching import dim_reduction, adopted_nearest_classifier
from PerformanceEvaluation import crr, roc
from FeatureExtraction import M1,Gabor, block, features
from tabulate import tabulate

def main():
    features_train=[]
    labels_train=[]
    # process all images for training
    for image in sorted(glob.glob('CASIA Iris Image Database (version 1.0)/*/1/*.bmp')):
        sample=image[40:43]
        label=int(sample)
        #print(image)
        image_train=cv2.imread(image)

        normal_train_img = normalization(image_train)
        enhanced_train_img=enhancement(normal_train_img)
        #train_feature=features(enhanced_train_img)
        train_feature=features(enhanced_train_img)
        assert len(train_feature)==1536
        features_train.append(train_feature)
        labels_train.append(label)
    features_train=np.array(features_train)
    labels_train=np.array(labels_train)
    
    # process all images for test
    features_test=[]
    labels_test=[]
    for image_t in sorted(glob.glob('CASIA Iris Image Database (version 1.0)/*/2/*.bmp')):
        sample=image_t[40:43]
        test_label=int(sample)
        #print(image_t)
        image_test=cv2.imread(image_t)

        normal_test_img = normalization(image_test)
        enhanced_test_img=enhancement(normal_test_img)
        #test_feature=features(enhanced_test_img)
        test_feature=features(enhanced_test_img)
        assert len(test_feature)==1536
        features_test.append(test_feature)
        labels_test.append(test_label)
    features_test=np.array(features_test)
    labels_test=np.array(labels_test)
    
    # get evaluations
    x,y,z, L1, L2, L3 = adopted_nearest_classifier(features_train,features_test,1536)
    a,b,c, _,_,_ = adopted_nearest_classifier(features_train,features_test,1300)
    x1,y1,z1 = crr(x,y,z,labels_train, labels_test)
    a,b,c = crr(a,b,c,labels_train, labels_test)
    print(tabulate([['L1 distance measure',x1 ,a],['L2 distance measure', y1,b], ['Cosine similarity measure', z1,c]], 
                   headers=['Similartiy measure', 'Original feature set',"Reduced feature set"]))
    
    com = np.array([30,  60,  80, 100, 120, 140, 160, 180, 200, 220])
    
    crr_l = []
    # figure 10
    for comp in com:
        #print(comp)
        x,y,z, L1, L2, L3 = adopted_nearest_classifier(features_train,features_test,comp)
        x1,y1,z1 = crr(x,y,z,labels_train, labels_test)
        crr_l.append(z1)
        
    plt.plot(com,crr_l,marker="*")
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('figure_12.png')
    plt.show()
    
    # table 4
    threshold = np.arange(0.04,0.1,0.001)
    fmrs, fnmrs = roc(threshold, labels_train, labels_test, z, L3)

    print(tabulate([[threshold[20],fmrs[20] ,fnmrs[20]], [threshold[40],fmrs[40] , fnmrs[40]],
                    [threshold[50],fmrs[50] ,fnmrs[50]]],
                   headers=['Threshold', 'False match rate(%)',"False non-match rate(%)"]))
    
    # Roc curve
    plt.plot(fmrs,fnmrs)
    plt.xlabel('False Match Rate(%)')
    plt.ylabel('False Non-match Rate(%)')
    plt.title('ROC')
    plt.savefig('roc_curve.png')
    plt.show()
    
    
main()