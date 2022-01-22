import numpy as np
from .GeneticOpertors import CrossoverMethod, SelectionMethod
from itertools import repeat

class GA():
    def __init__(self, 
                objectiveFunc, 
                popSize,
                maxGene,
                objective = "min",
                mutateProb = 0.001, 
                crossoverProb = 0.9, 
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
        self.maxGene = maxGene
        self.instanceLength = len(self.maxGene)
        self.history = []
        
        if crossoverMethod == "onePointCrossover":
            self.crossover = CrossoverMethod.onePointCrossover
        else:
            raise NotImplementedError

        if selectionMethod == "RWS":
            self.selectionMethod = SelectionMethod.RWS
        else:
            raise NotImplementedError
        

        self.fitVal = np.random.randint(low=0, high=1, size=(self.popSize, ))
        self.population = np.random.randint(low=0, high=1, size=(self.popSize, self.instanceLength))
        for i in range (self.instanceLength):
            self.population[:,i] = np.random.randint(low=0, 
                                                     high=self.maxGene[i], 
                                                     size = (1, self.popSize))
        self.allTimeBestInstance = self.population[0]
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
        increase = np.random.randint(low=0, high = np.max(self.maxGene), 
                                    size=self.population.shape)
            #this way of random can be bias
        self.population = (self.population + mutate*increase)%self.maxGene

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
        self.history += [self.objectiveFunc(self.bestInstance())]
        if self.objective=="min" and self.objectiveFunc(self.bestInstance()) < self.objectiveFunc(self.allTimeBestInstance):
            self.allTimeBestInstance = self.bestInstance()
        if self.objective=="max" and self.objectiveFunc(self.bestInstance()) > self.objectiveFunc(self.allTimeBestInstance):
            self.allTimeBestInstance = self.bestInstance()

    def fit(self, numGen):
        for _ in repeat(None, numGen):
            self._nextGen()

    def bestInstance(self):
        return self.population[np.argmax(self.fitVal)]