def myStrategy(pastPriceVec, currentPrice, stockType):
	# Explanation of my approach:
	# 240 天為一期，每支股票的每一期有一組獨立的 RSI 參數
	# 此 RSI 參數在 search.ipynb 中以暴力搜索方式產生，下面有解釋參數搜索的範圍    
	# 1. Technical indicator used: RSI
	# 2. if 50 < RSI < high or RSI < low ==> buy
	#    if 50 > RSI > low or RSI > high ==> sell
	# 3. Modifiable parameters: high, low, and window size for RSI
	#    (in course PPT high is fixed to 80 and low is fixed to 20)
	# 4. Use exhaustive search to obtain these parameter values (as shown in bestParamByExhaustiveSearch.py)
	#    range of exhaustive search is : 5 <= window size <= 20, 60 <= high <= 90, 10 <= low <= 45
	import numpy as np
	# stockType='SPY', 'IAU', 'LQD', 'DSI'
	# Set parameters for different stocks
	paramSetting={'SPY': [{ 'high': 65 , 'low': 44 , 'windowSize': 18 },
{ 'high': 60 , 'low': 45 , 'windowSize': 14 },
{ 'high': 80 , 'low': 37 , 'windowSize': 13 },
{ 'high': 61 , 'low': 12 , 'windowSize': 11 },
{ 'high': 60 , 'low': 32 , 'windowSize': 12 },
{ 'high': 77 , 'low': 42 , 'windowSize': 18 },
{ 'high': 76 , 'low': 21 , 'windowSize': 14 },
{ 'high': 77 , 'low': 42 , 'windowSize': 20 },
{ 'high': 80 , 'low': 45 , 'windowSize': 14 },
{ 'high': 78 , 'low': 45 , 'windowSize': 15 },
{ 'high': 69 , 'low': 40 , 'windowSize': 20 },
{ 'high': 69 , 'low': 40 , 'windowSize': 20 },
{ 'high': 78 , 'low': 39 , 'windowSize': 18 },
{ 'high': 78 , 'low': 39 , 'windowSize': 18 },
{ 'high': 78 , 'low': 37 , 'windowSize': 16 },
{ 'high': 78 , 'low': 35 , 'windowSize': 16 }],
					'IAU': [{ 'high': 78 , 'low': 44 , 'windowSize': 19 },
{ 'high': 75 , 'low': 19 , 'windowSize': 11 },
{ 'high': 74 , 'low': 44 , 'windowSize': 15 },
{ 'high': 67 , 'low': 34 , 'windowSize': 20 },
{ 'high': 75 , 'low': 45 , 'windowSize': 15 },
{ 'high': 69 , 'low': 41 , 'windowSize': 17 },
{ 'high': 80 , 'low': 45 , 'windowSize': 17 },
{ 'high': 62 , 'low': 22 , 'windowSize': 16 },
{ 'high': 60 , 'low': 31 , 'windowSize': 20 },
{ 'high': 63 , 'low': 36 , 'windowSize': 14 },
{ 'high': 70 , 'low': 12 , 'windowSize': 14 },
{ 'high': 70 , 'low': 31 , 'windowSize': 14 },
{ 'high': 68 , 'low': 31 , 'windowSize': 10 },
{ 'high': 80 , 'low': 45 , 'windowSize': 17 },
{ 'high': 71 , 'low': 44 , 'windowSize': 11 },
{ 'high': 80 , 'low': 45 , 'windowSize': 17 },],
					'LQD': [{ 'high': 63 , 'low': 38 , 'windowSize': 10 },
{ 'high': 79 , 'low': 32 , 'windowSize': 7 },
{ 'high': 67 , 'low': 42 , 'windowSize': 7 },
{ 'high': 60 , 'low': 15 , 'windowSize': 7 },
{ 'high': 78 , 'low': 24 , 'windowSize': 20 },
{ 'high': 78 , 'low': 45 , 'windowSize': 20 },
{ 'high': 66 , 'low': 35 , 'windowSize': 9 },
{ 'high': 65 , 'low': 40 , 'windowSize': 15 },
{ 'high': 80 , 'low': 30 , 'windowSize': 11 },
{ 'high': 63 , 'low': 45 , 'windowSize': 18 },
{ 'high': 79 , 'low': 44 , 'windowSize': 9 },
{ 'high': 80 , 'low': 45 , 'windowSize': 9 },
{ 'high': 80 , 'low': 42 , 'windowSize': 17 },
{ 'high': 80 , 'low': 41 , 'windowSize': 17 },
{ 'high': 80 , 'low': 41 , 'windowSize': 17 },
{ 'high': 80 , 'low': 31 , 'windowSize': 8 }],
					'DSI': [{ 'high': 60 , 'low': 45 , 'windowSize': 16 },
{ 'high': 63 , 'low': 30 , 'windowSize': 20 },
{ 'high': 63 , 'low': 44 , 'windowSize': 20 },
{ 'high': 72 , 'low': 35 , 'windowSize': 17 },
{ 'high': 68 , 'low': 23 , 'windowSize': 16 },
{ 'high': 60 , 'low': 43 , 'windowSize': 20 },
{ 'high': 75 , 'low': 45 , 'windowSize': 18 },
{ 'high': 75 , 'low': 45 , 'windowSize': 15 },
{ 'high': 68 , 'low': 25 , 'windowSize': 10 },
{ 'high': 77 , 'low': 40 , 'windowSize': 20 },
{ 'high': 80 , 'low': 40 , 'windowSize': 19 },
{ 'high': 80 , 'low': 40 , 'windowSize': 19 },
{ 'high': 72 , 'low': 38 , 'windowSize': 16 },
{ 'high': 76 , 'low': 36 , 'windowSize': 16 }]}
	dataLen = len(pastPriceVec)
	windowSize = paramSetting[stockType][dataLen//240]['windowSize']
	high = paramSetting[stockType][dataLen//240]['high']
	low = paramSetting[stockType][dataLen//240]['low']

	action = 0
	if dataLen == 0:
		return 0
	if dataLen < windowSize:
		windowSize = dataLen
	rsi = RSI(pastPriceVec, windowSize)
	if 50 < rsi < high or rsi < low:
		action = 1
	elif low < rsi < 50 or rsi > high:
		action = -1
	else:
		action = 0
	return action

def RSI(pastPriceVec, windowSize):
	import numpy as np
	up_down = np.zeros(windowSize)
	for i in range(1,windowSize):
		up_down[i] = pastPriceVec[-windowSize + i] - pastPriceVec[-windowSize + i - 1]
	rsi = up = down = 0
	for i in up_down:
		if i > 0:
			up += i
		else:
			down -= i
	if up + down != 0:
		rsi = up/(up + down) * 100
	return rsi