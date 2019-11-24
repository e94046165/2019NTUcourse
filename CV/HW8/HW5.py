#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Use the octogonal 3-5-5-5-3 kernel.

# Write programs which do gray scale morphological dilation, erosion, opening, and closing on a gray scale image.

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
#用 pyplot 使 Img 顯示在 jupyter notebook 中

def dilate(img, kernel):
    img_dilation = np.zeros([len(img),len(img[0])],dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            maxx = 0
            for k in range(i-len(kernel)//2, i+len(kernel)//2+1):
                for l in range(j-len(kernel)//2, j+len(kernel)//2+1):
                    if kernel[k - i + len(kernel)//2][l - j + len(kernel)//2] == 1:
                        if 0 <= k <= len(img)-1 and 0 <= l <= len(img[0])-1:
                            if img[k][l] > maxx:
                                maxx = img[k][l]
            if maxx > 255:
                maxx = 255
            img_dilation[i][j] = maxx
    return  img_dilation



# In[29]:


def erode(img, kernel):
    img_erosion = np.zeros([len(img),len(img[0])],dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            minn = 255
            for k in range(i-len(kernel)//2, i+len(kernel)//2+1):
                for l in range(j-len(kernel)//2, j+len(kernel)//2+1):
                    if kernel[k - i + len(kernel)//2][l - j + len(kernel)//2] == 1:
                        if 0 <= k <= len(img)-1 and 0 <= l <= len(img[0])-1:
                            if img[k][l] < minn:
                                minn = img[k][l]
            if minn < 0:
                minn = 0
            img_erosion[i][j] = minn
    return  img_erosion


# In[30]:

# In[31]:


def opening(img, kernel):
    img_opening = erode(img, kernel)
    img_opening = dilate(img_opening, kernel)
    return img_opening


# In[36]:

# In[34]:


def closing(img, kernel):
    img_closing = dilate(img, kernel)
    img_closing = erode(img_closing, kernel)
    return img_closing
