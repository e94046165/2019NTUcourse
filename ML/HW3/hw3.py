#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from matplotlib import pyplot as plt

def getData(path):
    file1 = open(path,"r")
    line = "start"
    x = []
    y = []
    while 1: 
        line = file1.readline()
        if not line:
            break
        line = line.replace('\t',' ')
        line = line.replace('\n','')
        line = line.split(" ")
        x.append([1]+[float(j) for j in line[1:len(line) -1] ])# 1 for x0
        y.append(int(line[-1]))
    file1.close()
    x = np.array(x)
    y = np.array(y)
    return x, y


'''
Implement the fixed learning rate gradient descent algorithm for logistic regression. 
Run the algorithm with η=0.001 and T = 2000. 
What is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?
'''
def sigmoid(s):
    return 1/(1 + np.exp(-s))
def sign(num):
    if num>= 0:
        return 1
    else:
        return -1

def logistic(T, n, w0, x, y):
    w = w0.copy()
    w_record = []
    for i in range(T):
        w = w + n * gradient(w, x, y)
        w_record.append(w)
    return w_record

def gradient(w ,x, y):
    N = len(y)
    err = np.array([0]*len(w))
    for n in range(N):
        err = err + sigmoid(-1 * y[n] * w.dot(x[n]))*(y[n]*x[n])
    return err/N


# In[2]:


train_x, train_y = getData('hw3_train')
test_x, test_y = getData("hw3_test")
def errorRate(w, testx, testy):
    wrong = 0
    for i in range(len(testx)):
        if sign(w.dot(testx[i])) != testy[i]:
            wrong += 1
    return wrong/len(testx)


# In[3]:


def logisticSGD(T, n, w0, x, y):
    w = w0.copy()
    w_record = []
    for i in range(T):
        err = np.array([0]*len(w))
        err = err + sigmoid(-1 * y[i%len(x)] * w.dot(x[i%len(x)]))*(y[i%len(x)]*x[i%len(x)])
        w = w + n * err
        #print(w)
        w_record.append(w)
    return w_record


# In[4]:


T = 2000
n = 0.001
w0 = np.array([0]*len(train_x[0]))
w_0001_GD = logistic(T, n, w0, train_x, train_y)
w_0001_SGD = logisticSGD(T, n, w0, train_x, train_y)
print("eta = 0.001")
print("=====GD=====")
print("Ein =",errorRate(w_0001_GD[-1], train_x, train_y))
print("Eout =",errorRate(w_0001_GD[-1], test_x, test_y))
print("=====SGD=====")
print("Ein =",errorRate(w_0001_SGD[-1], train_x, train_y))
print("Eout =",errorRate(w_0001_SGD[-1], test_x, test_y))
print()

# In[5]:


T = 2000
n = 0.01
w0 = np.array([0]*len(train_x[0]))
w_001_GD = logistic(T, n, w0, train_x, train_y)
w_001_SGD = logisticSGD(T, n, w0, train_x, train_y)
print("eta = 0.01")
print("=====GD=====")
print("Ein =",errorRate(w_001_GD[-1], train_x, train_y))
print("Eout =",errorRate(w_001_GD[-1], test_x, test_y))
print("=====SGD=====")
print("Ein =",errorRate(w_001_SGD[-1], train_x, train_y))
print("Eout =",errorRate(w_001_SGD[-1], test_x, test_y))


# In[6]:


Ein_0001_GD = []
Ein_0001_SGD = []
for w in w_0001_GD:
    Ein_0001_GD.append(errorRate(w, test_x, test_y))
for w in w_0001_SGD:
    Ein_0001_SGD.append(errorRate(w, test_x, test_y))


# In[7]:


Ein_001_GD = []
Ein_001_SGD = []
for w in w_001_GD:
    Ein_001_GD.append(errorRate(w, test_x, test_y))
for w in w_001_SGD:
    Ein_001_SGD.append(errorRate(w, test_x, test_y))


# In[17]:


x = [i for i in range(2000)]
plt.plot(x, Ein_0001_SGD, '-', label='Ein_0001_SGD', markersize=1)
plt.plot(x, Ein_0001_GD, '-', label='Ein_0001_GD', markersize=1)
plt.legend()
plt.savefig('eta0001.png')
plt.show()


# In[16]:


x = [i for i in range(2000)]
plt.plot(x, Ein_001_SGD, '-', label='Ein_001_SGD', markersize=1)
plt.plot(x, Ein_001_GD, '-', label='Ein_001_GD', markersize=1)
plt.legend()
plt.savefig('eta001.png')
plt.show()


# In[ ]:





# In[ ]:




