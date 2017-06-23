# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        # must pass through the data self.max_iterations times

        bestResult = 0
        bestC = float("inf")
        for index3, C in enumerate(Cgrid):
            self.C = C

            # pass through the data so many times
            ite = 0
            while ite < self.max_iterations:
                print "C = ",self.C,"   iter = ",ite+1
                for index2, tData in enumerate(trainingData):
                    maxScore = -float("inf")
                    for index, label in enumerate(self.weights):
                        score = 0
                        for feature in self.features:
                            score += tData[feature] * self.weights[index][feature]
                        if score > maxScore:
                            maxScore = score
                            bestClass = index
                    correctClass = trainingLabels[index2]
                    if bestClass <> correctClass:
                        # find tau from (wy - wy*) . f + 1  /  2f.f
                        numerator = 0
                        denominator = 0
                        for feature in self.features:
                            numerator += (self.weights[bestClass][feature] - self.weights[correctClass][feature]) *tData[feature]
                            denominator += tData[feature]**2

                        tau = (numerator + 1.0) / (2.0 * denominator)
                        tau = min(self.C, tau)

                        for feature in self.features:
                            self.weights[bestClass][feature] -= tData[feature] * tau
                            self.weights[correctClass][feature] += tData[feature]*tau
                ite += 1
            # testing complete, validate data
            result = 0
            for index2, vData in enumerate(validationData):
                
                maxScore = -float("inf")
                for index, label in enumerate(self.weights):
                    score = 0
                    for feature in self.features:
                        score += tData[feature] * self.weights[index][feature]
                    if score > maxScore:
                        maxScore = score
                        bestClass = index
                correctClass = validationLabels[index2]
                if bestClass == correctClass:
                    result += 1
            print "result = ",result
            #check if this is the best C
            if result == bestResult:
                if self.C < bestC:
                    bestC = self.C
                    bestWeights = self.weights
            if result > bestResult:
                bestResult = result
                bestC = self.C
                bestWeights = self.weights

        self.weights = bestWeights
        print "best C = ",bestC
        print "best result = ", bestResult
        return
        util.raiseNotDefined()

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


