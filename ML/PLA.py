#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
from matplotlib import pyplot as plt
file1 = open("hw1_6_train","r")
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
    x.append([float(j) for j in line ])
    


# In[ ]:





# In[62]:


w = [0, 0, 0, 0] #initialize w0

count = 0
while True:
    completed = True
    for i in range(20):
        Y = 0
        Y = np.dot(w, x[i][:4])
        if np.sign(Y) != x[i][-1]:
            for k in range(4):
                w[k] += x[i][-1]*x[i][k]
            completed = False 
    print(w)
    if completed == True:
        break
print(w)


# In[59]:


a = [1, 1, 1, 1]
b = [2, 2, 2, 2]
print(np.dot(a, b))


# In[ ]:




