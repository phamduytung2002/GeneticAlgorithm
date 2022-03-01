import numpy as np

"""
    input: fitVal: np.array of shape (popSize, )
    output: selectedIndex: np.array of shape (popSize, )
"""

def RWS(fitVal):
    relativeFitVal = fitVal/fitVal.sum()
    relativeFitVal = np.cumsum(relativeFitVal)
    selectedIndex = np.random.rand(fitVal.shape[0])
    selectedIndex = np.searchsorted(relativeFitVal, selectedIndex)
    return selectedIndex

def tournamentSelection(fitVal):
    raise NotImplementedError