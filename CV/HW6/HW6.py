#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('lena.bmp')
def imshow(img):
    plt.imshow(img, cmap='Greys_r',vmin = 0, vmax = 255)
    plt.show()
#change into binary image, threshold = 128 
img2 = np.zeros([len(img),len(img[0])],dtype=np.uint8)
for i in range(len(img)):
    for j in range(len(img[0])):
        if img[i][j][0] >= 128:
            img2[i][j] = 255
        else:
            img2[i][j] = 0

# Downsampling Lena from 512x512 to 64x64
img_64 = np.zeros([64,64],dtype=np.uint8)
for i in range(len(img_64)):
    for j in range(len(img_64)):
        img_64[i][j] = img2[i*8][j*8]
#imshow(img_64)

def h(b, c, d, e):
    if b == c:
        if d != b or e != b:
            return "q" 
        else:
            return "r"
    else:
        return "s"
def f(a1, a2, a3, a4):
    if a1 == a2 and a2 == a3 and a3 == a4 and a4 == "r":
        return 5
    else:
        a = 0
        for i in [a1, a2, a3, a4]:
            if i == "q":
                a += 1
        return a   
label = np.zeros([64,64],dtype=np.uint8)

for i in range(len(img_64)):
    for j in range(len(img_64[0])):
        if img_64[i][j] != 0:
            a1 = a2 = a3 = a4 = "s"
            b = img_64[i][j]
            # a1
            c = d = e = 0
            if j != len(img_64[0]) - 1 and i != 0:
                c = img_64[i][j+1]
                d = e = img_64[i-1][j+1]
                e = img_64[i-1][j]
            elif i != 0:
                e = img_64[i-1][j]
            elif j != len(img_64[0]) - 1:
                c = img_64[i][j+1]
            a1 = h(b, c, d, e)
            #a2
            c = d = e = 0
            if j != 0 and i != 0:
                c = img_64[i-1][j]
                d = e = img_64[i-1][j-1]
                e = img_64[i][j-1]
            elif i != 0:
                c = img_64[i-1][j]
            elif j != 0:
                e = img_64[i][j-1]  
            a2 = h(b, c, d, e)
            #a3
            c = d = e = 0
            if j != 0 and i != len(img_64) - 1:
                c = img_64[i][j-1]
                d = e = img_64[i+1][j-1]
                e = img_64[i+1][j]
            elif i != len(img_64) - 1:
                e = img_64[i+1][j]
            elif j != 0:
                c = img_64[i][j-1]  
            a3 = h(b, c, d, e)
            #a4
            c = d = e = 0
            if j != len(img_64[0]) - 1 and i != len(img_64) - 1:
                c = img_64[i+1][j]
                d = e = img_64[i+1][j+1]
                e = img_64[i][j+1]
            elif i != i != len(img_64) - 1:
                c = img_64[i+1][j]
            elif j != len(img_64[0]) - 1:
                e = img_64[i][j+1]  
            a4 = h(b, c, d, e)
            label[i][j] = f(a1, a2, a3, a4)


# In[5]:


for i in range(64):
    for j in range(64):
        if label[i][j] != 0:
            print(label[i][j],end="")
        else:
            print(" ",end="")
    print("")


# In[ ]:





# In[6]:

