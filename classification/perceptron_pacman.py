# perceptron_pacman.py
# --------------------
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # results stored in 0 and possible actions in 1
                # initialise best score to infinity to beat
                bestScore = -float("inf")
                # cycle through available actions in trainingData
                for action in trainingData[i][1]:
                    # score = 0
                    # calculate the score for the new state using trainingData and weights
                    for feature in self.features:
                        # trainingData[i][0] is a list of actions and corresponding
                        # foodcounts
                        # foodCount gets lower as Pacman does well
                        # should choose the result with the lowest score?
                        if feature <> "foodCount":
                            print feature
                        score = (100-trainingData[i][0][action][feature]) * self.weights[action]
                    # if it's the lowest score yet, update best score and best action
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                # update weights
                # print weights every 10 iterations in training data
                if i <10:
                    print "---------"
                    print self.weights
                    print trainingData[i][0]
                # get the correct action from trainingLabels
                correctAction = trainingLabels[i]
                # modify the action taken according to the feature value
                # remove from incorrect action
                if bestAction <> correctAction:
                    self.weights[bestAction] -= (100-trainingData[i][0][bestAction][feature])
                if i < 10:
                    print "bA", bestAction
                    print "bS", bestScore
                    print "cA", correctAction
                    print self.weights
                # add to the correct action
                if bestAction <> correctAction:
                    self.weights[correctAction] += (100-trainingData[i][0][correctAction][feature])

                if i < 10:
                    print self.weights
                                


                
                #util.raiseNotDefined()
