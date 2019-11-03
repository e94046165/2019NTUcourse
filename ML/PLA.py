#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from matplotlib import pyplot as plt

#================================ 讀入Dataset
file1 = open("hw1_6_train","r")
line = "start"
x = []

while 1:
    line = file1.readline()
    if not line:
        break
    line = line.replace('\t',' ')
    line = line.replace('\n','')
    line = line.split(" ")
    x.append([1]+[float(j) for j in line ])# 1 for x0
file1.close()


# In[2]:
#==================== 定義 PLA function

def pla(dataset):
    order = []
    for i in range(len(dataset)):
        order.append(i)
    random.shuffle(order)
    update = 0
    w = [0, 0, 0, 0, 0]
    while True:#迴圈直到 weight w 能完美分割 dataset
        change = 0
        for i in order:
            Y = np.dot(w, dataset[i][:len(w)])
            if np.sign(Y) != dataset[i][-1]: 
                for k in range(len(w)):
                    w[k] += dataset[i][-1]*dataset[i][k] # 更新 w 為 w + y*x
                change = 1
                update += 1 #紀錄更新 w 次數
        if change == 0:
            break
    return w ,update


# In[4]:


histogram = []
for i in range(1126):
    w, update = pla(x)
    histogram.append(update)
print(np.mean(histogram))
plt.hist(histogram)
plt.savefig("histogram_6.png",format="png")

