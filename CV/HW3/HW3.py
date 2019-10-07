import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
#用 pyplot 使 Img 顯示在 jupyter notebook 中
def imshow(img):
    plt.imshow(img, cmap='Greys_r',vmin = 0, vmax = 255)
    plt.show()
#3-a histogram
histogram = [0]*256
for i in range(len(img)):
    for j in range(len(img)):
        histogram[img[i][j]] += 1
imshow(img)
plt.bar(range(len(histogram)) , histogram) 
plt.xlabel('Colors')
plt.ylabel('Pixel')
plt.title('histogram')
plt.show()

#3-b
img2 = img.copy()
histogram2 = [0]*256
for i in range(len(img2)):
    for j in range(len(img2[0])):
        img2[i][j] = int(img2[i][j]/3)
        histogram2[img2[i][j]] += 1
imshow(img2)
plt.bar(range(len(histogram2)) , histogram2) 
plt.xlabel('Colors')
plt.ylabel('Pixel')
plt.title('histogram2')
plt.show()

img3 = img2.copy()
cdf = [0]*256
cdf[0] = histogram2[0]
histogram3 = [0]*256
for i in range(1,256):
    cdf[i] = cdf[i-1] + histogram2[i]
for i in range(len(img3)):
    for j in range(len(img3[0])):
        img3[i][j] = int((cdf[img3[i][j]] - min(cdf))/(len(img3)*len(img3[0]) - min(cdf))*255)
        histogram3[img3[i][j]] += 1
imshow(img3)
plt.bar(range(len(histogram3)) , histogram3) 
plt.xlabel('Colors')
plt.ylabel('Pixel')
plt.title('histogram3')
plt.show()
