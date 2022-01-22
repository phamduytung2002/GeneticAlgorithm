from GA import GA
import numpy as np
import time

key = np.array([1,2,3,4,5])

def testEval(x):
    return (x==key).sum()

startTime = time.time()
testing = GA(testEval, 500, objective="max")
testing.fit(100)
endTime = time.time()
print(testing.BestInstance())
print("time: {}".format(endTime-startTime))