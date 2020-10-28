# -*- coding: utf-8 -*-
"""
@author: Xue Xia (xx2338)
"""



import numpy as np
import math
from scipy import ndimage




def M1(x ,y, f):
    # the modulating function of defined filter
    m1 = np.cos(2*np.pi*f*math.sqrt(x**2 + y**2))
    return m1



def Gabor(x, y, f, delta_x, delta_y, i):
    if (i==1):
        M = M1(x, y, f)
    # We only use difined filter here
    else:
        M = M1(x, y, f)
    G = (1/(2*math.pi*delta_x*delta_y))*np.exp(-0.5*(x**2 / delta_x**2 + y**2 / delta_y**2)) * M
    return G



def block(f, delta_x, delta_y, i, size):
    # filter grid of size*size 
    w = np.zeros((size,size))
    for k in range(size):
        for j in range(size):
            w[k,j]=Gabor((-int(size/2)+j),(-int(size/2)+k),f,delta_x,delta_y,i)
    return w


def features(img):
    # filter parameter as in the paper
    filter1 = block(1/3,3,1.5,1,9)
    filter2 = block(1/4.5,4.5,1.5,1,9) 
    now = img[:48,:]
    
    # list to store the features
    f=[]
    #F1 = scipy.signal.convolve2d(now,filter1,mode='same')
    #F2 = scipy.signal.convolve2d(now,filter2,mode='same')
    F1 = ndimage.convolve(now, np.real(filter1), mode='wrap', cval=0)
    F2 = ndimage.convolve(now, np.real(filter2), mode='wrap', cval=0)
    
    # extract statistical features in each 8*8 small block 
    for i in range(6):
            for j in range(64):

                w1 = F1[(i*8):(i*8+8), (j*8):(j*8+8)]
                w2 = F2[(i*8):(i*8+8), (j*8):(j*8+8)]

                #1
                w_abs = np.absolute(w1)
                m = np.mean(w_abs)
                f.append(m)
                std = np.mean(np.absolute(w_abs-m))
                f.append(std)
                #2
                w_abs = np.absolute(w2)
                m = np.mean(w_abs)
                f.append(m)
                std = np.mean(np.absolute(w_abs-m))
                f.append(std)
                
    #length of 1536
    return f #length of 1536