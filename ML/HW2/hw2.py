#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


'''
def computeN(dvc, err, delta, NN):
    res = []
    for N in NN:
        tmp = 4*(math.pow(2*N, dvc))*math.exp(-1/8 * (err**2) * N)
        res.append('When sample N = ' + str(N) + ':    ' + str(tmp))
    for j in res:
        print(j)

N = [400000, 420000, 440000, 460000, 480000]
computeN(10, 0.05, 0.05, N)
'''
def sign(x): #用來判斷正負
    if x > 0:
        return 1
    else:
        return -1
    
#generate data by a uniform distribution in [a, b] of size = N, 
#noise_P means the probability of noise(flips), which should in [0, 1]
def gen(a, b, N, noise_P):
    x = np.random.uniform(a, b, N)
    noise = np.random.uniform(0, 1, N)
    y = [0]*len(x)
    flip_count = 0
    for i in range(N):
        if noise[i] <= noise_P:
            y[i] = sign(x[i])*(-1)
            flip_count += 1
        else:
            y[i] = sign(x[i])
    #print(flip_count)
    return x, y
def decisionStump(a, b, N, noise_P):
    theta = [-1]
    Ein_min = 1
    data_x, data_y = gen(a, b, N, noise_P)
    for i in range(len(data_x)-1):
        theta.append((data_x[i]+data_x[i+1])/2)
    theta.append(1)
    #print(data_x)
    #print(theta)
    # First, set s = 1, find the minimum Ein when s = 1, and update Ein_min and theta_best
    s = 1
    theta_best = 0
    for i in theta:
        wrong = 0
        for j in range(len(data_x)):
            if data_y[j]*sign(data_x[j] - i) != 1:
                wrong += 1
        if wrong/len(data_x) < Ein_min:
            Ein_min = wrong/len(data_x)
            theta_best = i
    # Then set s = -1, if there's any Ein less than the Ein_min before, update Ein_min, s and theta_best
    for i in theta:
        wrong = 0
        for j in range(len(data_x)):
            if -1*data_y[j]*sign(data_x[j] - i) != 1:
                wrong += 1
        if wrong/len(data_x) < Ein_min:
            s = -1
            Ein_min = wrong/len(data_x)
            theta_best = i
    Eout = 0.5 + 0.3*s*(abs(theta_best) - 1)
    return Ein_min, Eout
#7 
Ein7 = []
Eout7 = []
repeat = 1000
for i in range(repeat):
    #print(i)
    a, b= decisionStump(-1, 1, 20, 0.2)
    Ein7.append(a)
    Eout7.append(b)
print(sum(Ein7)/len(Ein7), sum(Eout7)/len(Eout7))


# In[ ]:


#8
Ein8 = []
Eout8 = []
repeat = 1000
for i in range(repeat):
    #print(i)
    a, b= decisionStump(-1, 1, 2000, 0.2)
    Ein8.append(a)
    Eout8.append(b)


# In[ ]:


print(sum(Ein8)/len(Ein7), sum(Eout8)/len(Eout7))

