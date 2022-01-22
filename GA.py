import numpy as np
import GeneticOpertors
from itertools import repeat

class GA():
    def __init__(self, 
                objectiveFunc, 
                popSize,
                objective = "min",
                mutateProb = 0.001, 
                crossoverProb = 0.9, 
                instanceLength = 5,
                lowBound = 0,
                highBound = 9,
                crossoverMethod = "onePointCrossover",
                selectionMethod = "RWS"):
    
        """
            objectiveFunc: a function take an instance and return its objective value
                np.array of shape (popSize, ) -> any type of number
            popSize: int, number of instances in the population
            objective: "min" if you want to minimize the objective function
                       "max" if you want to maximize the objective function
            mutateProb: float in [0,1], the probability of mutation operator
            crossoverProb: float in [0,1], the probility of crossover operator
            instanceLength: int, number of genes in one chromosome
            lowBound, highBound: int, the range of any genes is [lowBound, highBound]
            crossoverMethod: in this version only accept "onePointCrossover"
            selectionMethod: in this version only accept "RWS"
        """

        self.objectiveFunc = objectiveFunc

        if objective not in ["max", "min"]:
            raise TypeError('Only "max" and "min" are allowed for objective')
        else:
            self.objective = objective

        self.mutateProb = mutateProb
        self.crossoverProb = crossoverProb
        self.popSize = popSize
        self.lowBound = lowBound
        self.highBound = highBound 
        self.rangeLength = highBound+1-lowBound
        self.instanceLength = instanceLength
        
        if crossoverMethod == "onePointCrossover":
            self.crossover = GeneticOpertors.CrossoverMethod.onePointCrossover
        else:
            raise NotImplementedError

        if selectionMethod == "RWS":
            self.selectionMethod = GeneticOpertors.SelectionMethod.RWS
        else:
            raise NotImplementedError
        

        self.fitVal = np.ones((self.popSize, ))
        self.population = np.random.randint(low=self.lowBound, 
                                            high=self.highBound+1, 
                                            size=(self.popSize, self.instanceLength))
        self._eval()
    
    def _eval(self):
        for i in range(self.popSize):
            self.fitVal[i] = self.objectiveFunc(self.population[i])
        if self.objective == "min":
            self.fitVal = np.max(self.fitVal)-self.fitVal

    def _mutate(self):
        mutate = np.random.choice(range(2), 
                                  size=self.population.shape, 
                                  p=[1-self.mutateProb, self.mutateProb])
        increase = np.random.choice(range(self.lowBound,self.highBound+1), 
                                    size=self.population.shape)
        self.population = (self.population + mutate*increase)%self.rangeLength

    def _parentSelection(self):
        selectedIndex = self.selectionMethod(self.fitVal)
        self.population = self.population[selectedIndex]
        np.random.shuffle(self.population)

    def _crossoverPop(self):
        for i in range(0, self.popSize, 2):
            self.population[i], self.population[i+1] = self.crossover(self.population[i], 
                                                                      self.population[i+1],
                                                                      crossoverProb = self.crossoverProb)

    def _nextGen(self):
        self._parentSelection()
        self._crossoverPop()
        self._mutate()
        self._eval()

    def fit(self, numGen):
        for _ in repeat(None, numGen):
            self._nextGen()

    def BestInstance(self):
        return self.population[np.argmax(self.fitVal)]