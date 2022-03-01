import numpy as np

"""
        input:  instanceA, instanceB: np.array of shape (instanceLength, )
                crossoverProb: float in [0,1]
        output: instanceA, instanceB: np.array of shape (instanceLength, )
"""

def onePointCrossover(instanceA, instanceB, crossoverProb):
        if np.random.rand() < crossoverProb:
            pos = np.random.randint(low=0, high=instanceA.shape[0]-1)
            instanceA[pos:], instanceB[pos:] = instanceB[pos:], instanceA[pos:]

        return instanceA, instanceB

def multiplePointCrossover(instanceA, instanceB, crossoverProb):
        raise NotImplementedError

def uniformCrossover(instanceA, instanceB, crossoverProb):
        raise NotImplementedError