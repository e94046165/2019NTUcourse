#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np

def myOptimAction(priceMat, transFeeRate):
    # Explanation of my approach:
    # Using DP to find the optimal solution
    # Def: 
    #      cashMax[day] = max(for stock in range(stockCount):
    #                           stockMaxHolding[day-1][stock]*(1 - transFeeRate)*priceMat[day][stock]),
    #                       , Maxcash[day - 1])
    #      stockMaxHolding[day][stock] = max(stockMaxHolding[day - 1][stock] 
    #                                     ,cashMax[day]*(1 - transFeeRate)/priceMat[day][stock])
    #以cashMax,stockMaxHolding 找出買賣點：
    #　　由後往前看,第一次出現　cashMax[day] != cashMax[day - 1]　即為最後一個賣點
    #  從賣點由後往前看出現　stockMaxHolding[day][hold] != stockMaxHolding[day-1][hold] 為買點
    #　從買點由後往前看出現　cashMax[day] != cashMax[day - 1]　即為最後買點
    #  
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    #print(dataLen)
    cashMax = np.zeros(dataLen-1)
    cashMax[0] = 1000
    stockMaxHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings

    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    # user definition

    for day in range(dataLen - 1):
        action = []
        if day > 0:
            cash_best = cashMax[day-1]
            hold = -1
            for stock in range(stockCount):
                if cash_best < stockMaxHolding[day-1][stock]*(1 - transFeeRate)*priceMat[day][stock]:
                    cash_best = stockMaxHolding[day-1][stock]*(1 - transFeeRate)*priceMat[day][stock]
            cashMax[day] = cash_best    
            for stock in range(stockCount):   
                if stockMaxHolding[day - 1][stock] < cashMax[day]*(1 - transFeeRate)/priceMat[day][stock]:
                    stockMaxHolding[day][stock] = cashMax[day]*(1 - transFeeRate)/priceMat[day][stock]
                else:
                    stockMaxHolding[day][stock] = stockMaxHolding[day - 1][stock]
    day = dataLen - 2
    while day != 0:
        hold = -1
        #sell
        while True:
            if cashMax[day] != cashMax[day - 1]:
                break
            else:
                day -= 1
        if day > 0:
            for stock in range(stockCount):
                if stockMaxHolding[day-1][stock]*(1 - transFeeRate)*priceMat[day][stock] == cashMax[day]:
                    action = [day, stock, -1, cashMax[day]/(1-transFeeRate)]
                    actionMat.append(action)
                    hold = stock
                    break
        #buy
        while True:
            if stockMaxHolding[day][hold] != stockMaxHolding[day-1][hold]:
                break
            else:
                if day == 0:
                    break
                day -= 1
        if day > 0:
            action = [day, -1, hold, cashMax[day]]
            actionMat.append(action)
    return actionMat[::-1]





