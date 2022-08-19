import sys
sys.path.append("")
from GA.GA import GA
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
import logging

"""
	Test with Traveling Saleman Problem:
		Input: an adjacency of a weighted graph
		Output: A cycle goes through every vertex with as small the sum of edges' weight as possible
"""

parser = argparse.ArgumentParser(description="TSP")
parser.add_argument("--crossover", "-c", type=np.float32, default=0.8, metavar='', help="Crossover probability")
parser.add_argument("--mutation", "-m", type=np.float32, default=0.01, metavar='', help="Mutation probability")
parser.add_argument("--population", "-p", type=np.int64, default=50, metavar='', help="Population size")
parser.add_argument("--generations", "-g", type=np.int64, default=10, metavar='', help="Number of generations")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('test/result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
			 crossoverProb=args.crossover,
			 mutateProb=args.mutation,
	     	 popSize = args.population, 
	     	 maxGene = np.arange(n-1, 1, -1),
	     	 objective="min")
TSP.fit(args.generations)
endTime = time.time()
logger.info(f'''
    -c {args.crossover} -m {args.mutation} -p {args.population} -g {args.generations}
    Best value: {TSP.allTimeBestVal}, running time: {endTime-startTime}
    ''')
plt.plot(TSP.history)
plt.show()