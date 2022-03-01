from GA.GA import GA
import numpy as np
import time
import matplotlib.pyplot as plt

"""
	Test with Traveling Saleman Problem:
		Input: an adjacency of a weighted graph
		Output: A cycle goes through every vertex with as small the sum of edges' weight as possible
"""

data = [[9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
    	[3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
    	[5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
   		[48, 48, 74, 9999, 0, 6, 6, 12 , 12, 48, 48, 48, 48, 74, 6, 6, 12],
   		[48, 48, 74, 0, 9999, 6, 6, 12 , 12, 48, 48, 48, 48, 74, 6, 6, 12],
    	[8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
    	[8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
		[5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0],
    	[5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0],
    	[3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3 , 0, 3 , 8, 8, 5],
    	[3, 0, 3, 48, 48, 8, 8, 5, 5 , 0, 9999, 3 , 0, 3 , 8, 8, 5],
		[0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5 , 8, 8, 5],
    	[3, 0, 3, 48, 48, 8, 8, 5, 5 , 0, 0, 3, 9999, 3 , 8, 8, 5],
		[5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24],
		[8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8],
    	[8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8],
    	[5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999]]

def objectiveFunc(x):
	objectiveVal = 0
	traverseList = [0]
	cityList = list(range(1, x.shape[0]+2))
	for i in x:
		traverseList += [cityList[i]]
		cityList.pop(i)
	traverseList += cityList + [0]
	for i in range(len(traverseList)-1):
		objectiveVal += data[traverseList[i]][traverseList[i+1]]
	return objectiveVal

startTime = time.time()
n = len(data)
TSP = GA(objectiveFunc = objectiveFunc, 
			 crossoverProb=0.99,
			 mutateProb=0.01,
	     	 popSize = 400, 
	     	 maxGene = np.arange(n-1, 1, -1),
	     	 objective="min")
TSP.fit(200)
endTime = time.time()
print(TSP.allTimeBestVal)
print("time: {}".format(endTime-startTime))
plt.plot(TSP.history)
plt.show()