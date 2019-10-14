def myStrategy(pastPriceVec, currentPrice, stockType):
	# Explanation of my approach:
	# 1. Technical indicator used: RSI
	# 2. if 50 < RSI < high or RSI < low ==> buy
	#    if 50 > RSI > low or RSI > high ==> sell
	# 3. Modifiable parameters: high, low, and window size for RSI
	#    (in course PPT high is fixed to 80 and low is fixed to 20)
	# 4. Use exhaustive search to obtain these parameter values (as shown in bestParamByExhaustiveSearch.py)
	#    range of exhaustive search is : 5 <= window size <= 30, 51 <= high <= 100, 0 <= low <= 49
	import numpy as np
	# stockType='SPY', 'IAU', 'LQD', 'DSI'
	# Set parameters for different stocks
	paramSetting={'SPY': {'high':87, 'low':40, 'windowSize':20},
					'IAU': {'high':83, 'low':41, 'windowSize':25},
					'LQD': {'high':98, 'low':44, 'windowSize':9},
					'DSI': {'high':85, 'low':42, 'windowSize':20}}
	windowSize = paramSetting[stockType]['windowSize']
	high = paramSetting[stockType]['high']
	low = paramSetting[stockType]['low']
	
	action = 0
	dataLen = len(pastPriceVec)
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