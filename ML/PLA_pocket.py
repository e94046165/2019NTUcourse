#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import random
from matplotlib import pyplot as plt
file1 = open("hw1_7_train","r")
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

file1 = open("hw1_7_test","r")
line = "start"

x_test = []

while 1:
    line = file1.readline()
    if not line:
        break
    line = line.replace('\t',' ')
    line = line.replace('\n','')
    line = line.split(" ")
    x_test.append([1]+[float(j) for j in line ])# 1 for x0
file1.close()


# In[73]:


def pla_pocket(dataset, iteration):
    order = []
    for i in range(len(dataset)):
        order.append(i)
    
    random.shuffle(order)
    update = 0
    w_best = [0, 0, 0, 0, 0]
    w_tmp = w_best
    err_w = len(dataset)
    while update < iteration:
        change = 0
        for i in order:
            Y = np.dot(w_tmp, dataset[i][:len(w_best)])
            if np.sign(Y) != dataset[i][-1]:
                for k in range(len(w_best)):
                    w_tmp[k] += dataset[i][-1]*dataset[i][k]
                change = 1
                update += 1
                err_wtmp = 0
                for j in range(len(dataset)):
                    if np.sign(np.dot(w_tmp, dataset[j][:len(w_best)])) != dataset[j][-1]:
                        err_wtmp += 1
                #print(err_w, err_wtmp)
                if err_wtmp < err_w:
                    w_best = w_tmp.copy()#why this copy() metters?
                    err_w = err_wtmp
                    #print("change!")
            if update == iteration:
                break
        if change == 0:
            print("Only update:", update, "times, finished")
            break
    #print(w_best == w_tmp)
    return w_best


# In[75]:


def err_rate(dataset, w):
    err = 0
    for i in range(len(dataset)):
        if np.sign(np.dot(w, dataset[i][:len(w)])) != dataset[i][-1]:
            err += 1
    return err/len(dataset)


# In[77]:


record = []
for i in range(1126):
    w = pla_pocket(x, 100)
    record.append(err_rate(x_test, w))
print(np.mean(record))


# In[93]:


plt.hist(record, [i*0.01 for i in range(0, 50, 1)])
plt.savefig("histogram_7.png",format="png")

