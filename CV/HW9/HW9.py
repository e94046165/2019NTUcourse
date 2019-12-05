#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2 
import numpy as np
import math
from matplotlib import pyplot as plt
img = cv2.imread('lena.bmp', 0)
def imshow(img):
    plt.imshow(img, cmap='Greys_r',vmin = 0, vmax = 255)
    plt.show()
def padding(img, size):# padding each edge with size//2
    img_padding = np.zeros(( len(img)+ size - 1, len(img[0]) + size - 1), dtype = np.int)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img_padding[i+size//2][j+size//2] = img[i][j]
    return img_padding


# In[98]:


#a) Roberts operator (threshold: 12)
def robertsOp(img, threshold):
    img_padding = padding(img, 3)
    img_robert = img.copy()
    for i in range(len(img_robert)):
        for j in range(len(img_robert[0])):
            r1 = (-1)*img_padding[i+1][j+1] + img_padding[i+2][j+2]
            r2 = img_padding[i+2][j+1] + (-1)*img_padding[i+1][j+2]
            if math.sqrt(r1**2 + r2**2) >= threshold:
                img_robert[i][j] = 0
            else:
                img_robert[i][j] = 255
    return img_robert
img_robert = robertsOp(img, 12)
#imshow(img_robert)
cv2.imwrite("robert_12.bmp", img_robert)


# In[99]:


#b) Prewitt edge detector (threshold: 24)
def prewittOp(img, threshold):
    img_padding = padding(img, 3)
    img_prewitt= img.copy()
    for i in range(len(img_prewitt)):
        for j in range(len(img_prewitt[0])):
            p1 = (-1)*img_padding[i+1-1][j+1-1] + (-1)*img_padding[i+1-1][j+1] + (-1)*img_padding[i+1-1][j+1+1]
            p1 += img_padding[i+1+1][j+1-1] + img_padding[i+1+1][j+1] + img_padding[i+1+1][j+1+1]
            p2 = (-1)*img_padding[i+1-1][j+1-1] + (-1)*img_padding[i+1][j+1-1] + (-1)*img_padding[i+1+1][j+1-1]
            p2 += img_padding[i+1-1][j+1+1] + img_padding[i+1][j+1+1] + img_padding[i+1+1][j+1+1]
            if math.sqrt(p1**2 + p2**2) >= threshold:
                img_prewitt[i][j] = 0
            else:
                img_prewitt[i][j] = 255        
    return img_prewitt
img_prewitt_24 = prewittOp(img, 24)
#imshow(img_prewitt_24)
cv2.imwrite("prewitt.bmp", img_prewitt_24)


# In[100]:


# (c) Sobel's Edge Detector: theshold = 38
def sobelOp(img, threshold):
    img_padding = padding(img, 3)
    img_sobel = img.copy()
    for i in range(len(img_sobel)):
        for j in range(len(img_sobel[0])):
            s1 = (-1)*img_padding[i+1-1][j+1-1] + (-2)*img_padding[i+1-1][j+1] + (-1)*img_padding[i+1-1][j+1+1]
            s1 += img_padding[i+1+1][j+1-1] + 2*img_padding[i+1+1][j+1] + img_padding[i+1+1][j+1+1]
            s2 = (-1)*img_padding[i+1-1][j+1-1] + (-2)*img_padding[i+1][j+1-1] + (-1)*img_padding[i+1+1][j+1-1]
            s2 += img_padding[i+1-1][j+1+1] + 2*img_padding[i+1][j+1+1] + img_padding[i+1+1][j+1+1]
            if math.sqrt(s1**2 + s2**2) >= threshold:
                img_sobel[i][j] = 0
            else:
                img_sobel[i][j] = 255        
    return img_sobel
img_sobel = sobelOp(img, 38)
#imshow(img_sobel)
cv2.imwrite("sobel.bmp", img_sobel)


# In[101]:


# (d) Frei and Chen's Gradient Operator: threshold = 30
def f_and_gOp(img, threshold):
    img_padding = padding(img, 3)
    img_fg = img.copy()
    for i in range(len(img_fg)):
        for j in range(len(img_fg[0])):
            f1 = (-1)*img_padding[i+1-1][j+1-1] + (-1)*math.sqrt(2)*img_padding[i+1-1][j+1] + (-1)*img_padding[i+1-1][j+1+1]
            f1 += img_padding[i+1+1][j+1-1] + math.sqrt(2)*img_padding[i+1+1][j+1] + img_padding[i+1+1][j+1+1]
            f2 = (-1)*img_padding[i+1-1][j+1-1] + (-1)*math.sqrt(2)*img_padding[i+1][j+1-1] + (-1)*img_padding[i+1+1][j+1+1-1]
            f2 += img_padding[i+1-1][j+1+1] + math.sqrt(2)*img_padding[i+1][j+1+1] + img_padding[i+1+1][j+1+1]
            if math.sqrt(f1**2 + f2**2) >= threshold:
                img_fg[i][j] = 0
            else:
                img_fg[i][j] = 255        
    return img_sobel
img_fg = f_and_gOp(img, 38)
#imshow(img_fg)
cv2.imwrite("freiAndChen.bmp", img_fg)


# In[54]:


# (e) Kirsch's Compass Operator: threshold = 135
def KirschCompassOp(img, threshold):
    img_padding = padding(img, 3)
    img_kc = img.copy()
    k = [0]*8
    for i in range(len(img_kc)):
        for j in range(len(img_kc[0])):
            k[0] = (-3)*img_padding[i+1-1][j+1-1] + (-3)*img_padding[i+1][j+1-1] + 5*img_padding[i+1+1][j+1-1]+                   (-3)*img_padding[i+1-1][j+1] + 0 + 5*img_padding[i+1+1][j+1]+                   (-3)*img_padding[i+1-1][j+1+1] + (-3)*img_padding[i+1][j+1+1] + 5*img_padding[i+1+1][j+1+1]
            
            k[1] = (-3)*img_padding[i+1-1][j+1-1] + 5*img_padding[i+1][j+1-1] + 5*img_padding[i+1+1][j+1-1]+                   (-3)*img_padding[i+1-1][j+1] + 0 + 5*img_padding[i+1+1][j+1]+                   (-3)*img_padding[i+1-1][j+1+1] + (-3)*img_padding[i+1][j+1+1] + (-3)*img_padding[i+1+1][j+1+1]
            
            k[2] = 5*img_padding[i+1-1][j+1-1] + 5*img_padding[i+1][j+1-1] + 5*img_padding[i+1+1][j+1-1]+                   (-3)*img_padding[i+1-1][j+1] + 0 + (-3)*img_padding[i+1+1][j+1]+                   (-3)*img_padding[i+1-1][j+1+1] + (-3)*img_padding[i+1][j+1+1] + (-3)*img_padding[i+1+1][j+1+1]
            
            k[3] = 5*img_padding[i+1-1][j+1-1] + 5*img_padding[i+1][j+1-1] + (-3)*img_padding[i+1+1][j+1-1]+                   5*img_padding[i+1-1][j+1] + 0 + (-3)*img_padding[i+1+1][j+1]+                   (-3)*img_padding[i+1-1][j+1+1] + (-3)*img_padding[i+1][j+1+1] + (-3)*img_padding[i+1+1][j+1+1]
            
            k[4] = 5*img_padding[i+1-1][j+1-1] + (-3)*img_padding[i+1][j+1-1] + (-3)*img_padding[i+1+1][j+1-1]+                   5*img_padding[i+1-1][j+1] + 0 + (-3)*img_padding[i+1+1][j+1]+                   5*img_padding[i+1-1][j+1+1] + (-3)*img_padding[i+1][j+1+1] + (-3)*img_padding[i+1+1][j+1+1]
            
            k[5] = (-3)*img_padding[i+1-1][j+1-1] + (-3)*img_padding[i+1][j+1-1] + (-3)*img_padding[i+1+1][j+1-1]+                   5*img_padding[i+1-1][j+1] + 0 + (-3)*img_padding[i+1+1][j+1]+                   5*img_padding[i+1-1][j+1+1] + 5*img_padding[i+1][j+1+1] + (-3)*img_padding[i+1+1][j+1+1]
            
            k[6] = (-3)*img_padding[i+1-1][j+1-1] + (-3)*img_padding[i+1][j+1-1] + (-3)*img_padding[i+1+1][j+1-1]+                   (-3)*img_padding[i+1-1][j+1] + 0 + (-3)*img_padding[i+1+1][j+1]+                   5*img_padding[i+1-1][j+1+1] + 5*img_padding[i+1][j+1+1] + 5*img_padding[i+1+1][j+1+1]
            
            k[7] = (-3)*img_padding[i+1-1][j+1-1] + (-3)*img_padding[i+1][j+1-1] + (-3)*img_padding[i+1+1][j+1-1]+                   (-3)*img_padding[i+1-1][j+1] + 0 + 5*img_padding[i+1+1][j+1]+                   (-3)*img_padding[i+1-1][j+1+1] + 5*img_padding[i+1][j+1+1] + 5*img_padding[i+1+1][j+1+1]
            if max(k) >= threshold:
                img_kc[i][j] = 0
            else:
                img_kc[i][j] = 255
    return img_kc


# In[56]:


img_kc = KirschCompassOp(img, 135)
#imshow(img_kc)
cv2.imwrite("KirschCompass.bmp", img_kc)


# In[58]:


# (f) Robinson's Compass Operator: threshold = 43
def RobinsonCompassOp(img, threshold):
    img_padding = padding(img, 3)
    img_rc = img.copy()
    r = [0]*8
    for i in range(len(img_rc)):
        for j in range(len(img_rc[0])):
            r[0] = (-1)*img_padding[i+1-1][j+1-1] + (0)*img_padding[i+1][j+1-1] + 1*img_padding[i+1+1][j+1-1]+                   (-2)*img_padding[i+1-1][j+1]   +  0                          + 2*img_padding[i+1+1][j+1]+                   (-1)*img_padding[i+1-1][j+1+1] + (0)*img_padding[i+1][j+1+1] + 1*img_padding[i+1+1][j+1+1]
            
            r[1] = (0)*img_padding[i+1-1][j+1-1] + (1)*img_padding[i+1][j+1-1] + 2*img_padding[i+1+1][j+1-1]+                   (-1)*img_padding[i+1-1][j+1]   +  0                          + 1*img_padding[i+1+1][j+1]+                   (-2)*img_padding[i+1-1][j+1+1] + (-1)*img_padding[i+1][j+1+1] + 0*img_padding[i+1+1][j+1+1]
            
            r[2] = 1*img_padding[i+1-1][j+1-1] + 2*img_padding[i+1][j+1-1] + 1*img_padding[i+1+1][j+1-1]+                   (0)*img_padding[i+1-1][j+1] + 0 + (0)*img_padding[i+1+1][j+1]+                   (-1)*img_padding[i+1-1][j+1+1] + (-2)*img_padding[i+1][j+1+1] + (-1)*img_padding[i+1+1][j+1+1]
            
            r[3] = 2*img_padding[i+1-1][j+1-1] + 1*img_padding[i+1][j+1-1] + (0)*img_padding[i+1+1][j+1-1]+                   1*img_padding[i+1-1][j+1] + 0 + (-1)*img_padding[i+1+1][j+1]+                   (0)*img_padding[i+1-1][j+1+1] + (-1)*img_padding[i+1][j+1+1] + (-2)*img_padding[i+1+1][j+1+1]
            
            r[4] = 1*img_padding[i+1-1][j+1-1] + (0)*img_padding[i+1][j+1-1] + (-1)*img_padding[i+1+1][j+1-1]+                   2*img_padding[i+1-1][j+1] + 0 + (-2)*img_padding[i+1+1][j+1]+                   1*img_padding[i+1-1][j+1+1] + (0)*img_padding[i+1][j+1+1] + (-1)*img_padding[i+1+1][j+1+1]
            
            r[5] = (0)*img_padding[i+1-1][j+1-1] + (-1)*img_padding[i+1][j+1-1] + (-2)*img_padding[i+1+1][j+1-1]+                   1*img_padding[i+1-1][j+1] + 0 + (-1)*img_padding[i+1+1][j+1]+                   2*img_padding[i+1-1][j+1+1] + 1*img_padding[i+1][j+1+1] + (0)*img_padding[i+1+1][j+1+1]
            
            r[6] = (-1)*img_padding[i+1-1][j+1-1] + (-2)*img_padding[i+1][j+1-1] + (-1)*img_padding[i+1+1][j+1-1]+                   (0)*img_padding[i+1-1][j+1] + 0 + (0)*img_padding[i+1+1][j+1]+                   1*img_padding[i+1-1][j+1+1] + 2*img_padding[i+1][j+1+1] + 1*img_padding[i+1+1][j+1+1]
            
            r[7] = (-2)*img_padding[i+1-1][j+1-1] + (-1)*img_padding[i+1][j+1-1] + (0)*img_padding[i+1+1][j+1-1]+                   (-1)*img_padding[i+1-1][j+1] + 0 + 1*img_padding[i+1+1][j+1]+                   (0)*img_padding[i+1-1][j+1+1] + 1*img_padding[i+1][j+1+1] + 2*img_padding[i+1+1][j+1+1]
            if max(r) >= threshold:
                img_rc[i][j] = 0
            else:
                img_rc[i][j] = 255
    return img_rc


# In[59]:


img_rc = RobinsonCompassOp(img, 43)
#imshow(img_rc)
cv2.imwrite("RobinsonhCompass.bmp", img_rc)


# In[102]:


# (g) Nevatia-Babu 5x5 Operator: threshold = 12500
def NB5x5(img, threshold):
    p = padding(img, 5)
    img_nb = img.copy()
    n = [0]*6
    for i in range(len(img_nb)):
        for j in range(len(img_nb[0])):
            n[0] = 100*p[i+2-2][j+2-2] + 100*p[i+2-2][j+2-1] + 100*p[i+2-2][j+2] + 100*p[i+2-2][j+2+1] + 100*p[i+2-2][j+2+2]+                   100*p[i+2-1][j+2-2] + 100*p[i+2-1][j+2-1] + 100*p[i+2-1][j+2] + 100*p[i+2-1][j+2+1] + 100*p[i+2-1][j+2+2]+                   0*p[i+2][j+2-2] + 0*p[i+2][j+2-1] + 0*p[i+2][j+2 ] + 0*p[i+2][j+2+1] + 0*p[i+2][j+2+2]+                   (-100)*p[i+2+1][j+2-2] + (-100)*p[i+2+1][j+2-1] + (-100)*p[i+2+1][j+2] + (-100)*p[i+2+1][j+2+1] + (-100)*p[i+2+1][j+2+2]+                   (-100)*p[i+2+2][j+2+2] + (-100)*p[i+2+2][j+2+2] + (-100)*p[i+2+2][j+2+2] + (-100)*p[i+2+2][j+2+2] + (-100)*p[i+2+2][j+2+2]
            
            n[1] = 100*p[i+2-2][j+2-2] + 100*p[i+2-2][j+2-1] + 100*p[i+2-2][j+2] + 100*p[i+2-2][j+2+1] + 100*p[i+2-2][j+2+2]+                   100*p[i+2-1][j+2-2] + 100*p[i+2-1][j+2-1] + 100*p[i+2-1][j+2] + 78*p[i+2-1][j+2+1] + (-32)*p[i+2-1][j+2+2]+                   100*p[i+2][j+2-2] +  92*p[i+2][j+2-1] +   0*p[i+2][j+2  ] + (-92)*p[i+2][j+2 +1] + (-100)*p[i+2][j+2+2 ]+                    32*p[i+2+1][j+2-2] + (-78)*p[i+2+1][j+2-1] + (-100)*p[i+2+1][j+2] + (-100)*p[i+2+1][j+2+1] + (-100)*p[i+2+2][j+2+2]+                   (-100)*p[i+2+2][j+2-2] + (-100)*p[i+2+2][j+2-1] + (-100)*p[i+2+2][j+2] + (-100)*p[i+2+2][j+2+1] + (-100)*p[i+2+2][j+2+2]
            
            n[2] = 100*p[i+2-2][j+2-2] + 100*p[i+2-2][j+2-1] + 100*p[i+2-2][j+2] + 32*p[i+2-2][j+2+1] + (-100)*p[i+2-2][j+2+2]+                   100*p[i+2-1][j+2-2] + 100*p[i+2-1][j+2-1] +  92*p[i+2-1][j+2] + (-78)*p[i+2-1][j+2+1] + (-100)*p[i+2-1][j+2+2]+                   100*p[i+2-0][j+2-2] + 100*p[i+2-0][j+2-1]  +  0*p[i+2-0][j+2] + (-100)*p[i+2][j+2+1] + (-100)*p[i+2][j+2+2]+                   100*p[i+2+1][j+2-2] +  78*p[i+2+1][j+2-1]+(-92)*p[i+2+1][j+2] + (-100)*p[i+2+1][j+2+1] + (-100)*p[i+2+1][j+2+2]+                   100*p[i+2+2][j+2-2]+(-32)*p[i+2+2][j+2-1]+(-100)*p[i+2+2][j+2] + (-100)*p[i+2+2][j+2+1] + (-100)*p[i+2+2][j+2+2]

            n[3] = (-100)*p[i+2-2][j+2-2] + (-100)*p[i+2-2][j+2-1] + 0*p[i+2-2][j+2  ] + 100*p[i+2-2][j+2+1] + 100*p[i+2-2][j+2+2]+                   (-100)*p[i+2-1][j+2-2] + (-100)*p[i+2-1][j+2-1] + 0*p[i+2-1][j+2  ] + 100*p[i+2-1][j+2+1] + 100*p[i+2-1][j+2+2]+                   (-100)*p[i+2  ][j+2-2] + (-100)*p[i+2  ][j+2-1] + 0*p[i+2  ][j+2  ] + 100*p[i+2  ][j+2+1] + 100*p[i+2  ][j+2+2]+                   (-100)*p[i+2+1][j+2-2] + (-100)*p[i+2+1][j+2-1] + 0*p[i+2+1][j+2  ] + 100*p[i+2+1][j+2+1] + 100*p[i+2+1][j+2+2]+                   (-100)*p[i+2+2][j+2-2] + (-100)*p[i+2+2][j+2-1] + 0*p[i+2+2][j+2  ] + 100*p[i+2+2][j+2+1] + 100*p[i+2+2][j+2+2]
            
            n[4] = (-100)*p[i+2-2][j+2-2] + 32*p[i+2-2][j+2-1] + 100*p[i+2-2][j+2]       + 100*p[i+2-2][j+2+1] + 100*p[i+2-2][j+2+2]+                   (-100)*p[i+2-1][j+2-2] + (-78)*p[i+2-1][j+2-1] + 92*p[i+2-1][j+2]     + 100*p[i+2-1][j+2+1] + 100*p[i+2-1][j+2+2]+                   (-100)*p[i+2  ][j+2-2] + (-100)*p[i+2][j+2-1] + 0*p[i+2][j+2]         + 100*p[i+2][j+2+1] + 100*p[i+2][j+2+2]+                   (-100)*p[i+2+1][j+2-2] + (-100)*p[i+2-1][j+2-1] + (-92)*p[i+2+1][j+2] + 78*p[i+2+1][j+2+1] + 100*p[i+2+1][j+2+2]+                   (-100)*p[i+2+2][j+2-2] + (-100)*p[i+2-2][j+2-1] + (-100)*p[i+2+2][j+2] + (-32)*p[i+2+2][j+2+1] + 100*p[i+2+2][j+2+2]
                       
            n[5] = 100*p[i+2-2][j+2-2] + 100*p[i+2-2][j+2-1] + 100*p[i+2-2][j+2] + 100*p[i+2-2][j+2+1] + 100*p[i+2-2][j+2+2]+                   (-32)*p[i+2-1][j+2-2] + 78*p[i+2-1][j+2-1] + 100*p[i+2-1][j+2] + 100*p[i+2-1][j+2+1] + 100*p[i+2-1][j+2+2]+                   (-100)*p[i+2][j+2-2] + (-92)*p[i+2][j+2-1] + 0*p[i+2][j+2  ] + 92*p[i+2][j+2+1] + 100*p[i+2][j+2+2]+                   (-100)*p[i+2+1][j+2-2] + (-100)*p[i+2+1][j+2-1] + (-100)*p[i+2+1][j+2] + (-78)*p[i+2+1][j+2+1] + 32*p[i+2+1][j+2+2]+                   (-100)*p[i+2+2][j+2-2] + (-100)*p[i+2+2][j+2-1] + (-100)*p[i+2+2][j+2] + (-100)*p[i+2+1][j+2+1] + (-100)*p[i+2+2][j+2+2]
            if max(n) >= threshold:
                img_nb[i][j] = 0
            else:
                img_nb[i][j] = 255
    return img_nb            


# In[104]:


img_nb = NB5x5(img, 12500)
#imshow(img_nb)
cv2.imwrite("Nevatia-Babu.bmp", img_nb)


# In[ ]:




