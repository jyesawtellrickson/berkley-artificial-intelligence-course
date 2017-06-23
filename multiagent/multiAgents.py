# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        evalGhostPos = 0.0
        testGhostPos = 0.0
        
        for state in newGhostStates:
            # get distance to ghosts
            # want this to get very bad, very close
            # approaches zero as it gets bad
            testGhostPos = abs(newPos[1] - state.getPosition()[1]) + abs(newPos[0] - state.getPosition()[0]) + 1
            #print state.getPosition(),"------",newPos,"-----",testGhostPos
            evalGhostPos += -100.0 / testGhostPos**3
            
        # evalGhostPos = evalGhostPos

        #if evalGhostPos > -2:
        #    evalGhostPos = 0

        evalFoodPos = 0
        evalFoodCount = 0
        evalFoodEat = 0
        minFoodDist = 1000
        
        for food in newFood.asList():
            # calculate the net distance from food.
            # Is it better to pick just a few?
            # add each distance
            # must divide by the total count of food to make sure changes <=1
            evalFoodPos += ((newPos[1] - food[1])**2 + (newPos[0] - food[0])**2)**(1.0/2.0)
            evalFoodCount += 1.0
            if (newPos[1] - food[1]) + (newPos[0] - food[0]) == 0:
                evalFoodEat = 1
            foodDist = abs(newPos[1] - food[1]) + abs(newPos[0] - food[0])
            if foodDist < minFoodDist:
                minFoodDist = foodDist
            
        # evalFoodPos should be strongest when there aren't many foods left
        evalFoodPos = -evalFoodPos/(evalFoodCount+1)
        evalFoodPos = 1.0/(minFoodDist+1)
        # pacman will thrash when he gets stuck between an even area
        # if values are even, always head left?
        #if evalFoodEat == 1:
        #    evalFoodPos = 0
        #import random
        #evalFoodPos = evalFoodPos + 2*random.random()
        # using evalFoodPos will just put Pacman in the middle of all the food always.
        # let's try going to the nearest food
        # if action == "West" or action == "South":
        #     evalFoodPos = evalFoodPos + 1
                
        # evalFoodCount = -100.0/evalFoodCount
        
        evalFood = -10000.0*(len(newFood.asList())+1)

        evalScore = -100.0/(currentGameState.getScore()+0.5)
        
        myEvalFun = evalFoodPos -evalFoodCount + evalGhostPos #+ evalFoodCount#+ evalFoodPos

        # don't stop!!!
        if action == "Stop":
            myEvalFun = -1000
        #print action, "------", myEvalFun
        #print evalGhostPos,"----",evalFoodPos, "-----", action#, "-----", evalFoodCount #, "-----", myEvalFun

        return myEvalFun

        
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # define initial depth as 1
        depth = 0
        self.action = []
        
        # want to find the sucessors for pacman (1-4 options) and
        # the successors for ghosts (1-4 options * number of ghosts)

        # start by initiating the max function for the current state
        bestAction = self.value(gameState, 0, depth)

        return self.action
        
        util.raiseNotDefined()

    def value(self, gameState, agent, depth):
        """
            Function that solves for value of any given state using
            the minmax approach.
        """
        if depth == self.depth:
            result = self.evaluationFunction(gameState)
            return result
        
        if agent == 0:
            result = self.maxFunc(gameState, depth)
            #print "max",result
        else:
            result = self.minFunc(gameState, agent, depth)
            #print "min",result
        
        return result
            
    def minFunc(self, gameState, agent, depth):
        """
          Function that returns the minimum value of the successorStates.
          The min function applies purely to the Ghosts
        """
        # initiate value of inf
        minActionValue = float("inf")
        # check the number of agents
        numAgents = gameState.getNumAgents()-1
        # get legal actions for current active agent
        newActions = gameState.getLegalActions(agent)
        # for each action, get the new state
        for action in newActions:
            newState = gameState.generateSuccessor(agent, action)
            # if we are on the last agent, get values
            if agent == numAgents:
                # call the maxfunction and increase the depth
                actionValue = self.value(newState, 0, depth+1)
            # if not on the last agent, still need to check other ghost moves
            else:
                # action value will come from the minFunc of the next agent
                actionValue = self.value(newState, agent+1, depth)        
                # if the action value is smaller, choose this value

            if actionValue < minActionValue:
                minActionValue = actionValue
                
        # if there were no new actions, simply return the current state value
        if newActions == []:
            minActionValue = self.evaluationFunction(gameState)

        # returns the minimum value calculated across actions
        return minActionValue

    def maxFunc(self, gameState, depth):

        # initialise value of -inf
        # action Value should not be defined...
        maxActionValue = -float("inf")
        bestAction = ""
        # get legal actions to find new states
        newActions = gameState.getLegalActions(0)
        # for each possible action, get the states and their values
        for action in newActions:
            # for each action, calculate the sucessor state and value
            newState = gameState.generateSuccessor(0, action)
            # get value from minfunc, starting with ghost 1
            actionValue = self.value(newState, 1, depth)
            # if the actionValue is greater than the previous max, best option
            if actionValue > maxActionValue:
                maxActionValue = actionValue
                bestAction = action
        if newActions == []:
            maxActionValue = self.evaluationFunction(gameState)

        self.action = bestAction
        
        return maxActionValue
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # define initial depth as 1
        depth = 0
        self.action = []
        alpha = -float("inf")
        beta = float("inf")
        # want to find the sucessors for pacman (1-4 options) and
        # the successors for ghosts (1-4 options * number of ghosts)

        # start by initiating the max function for the current state
        # changed bestAction to bestScore
        bestScore = self.value(gameState, 0, depth, alpha, beta)
        return self.action
        
    def value(self, gameState, agent, depth, alpha, beta):
        """
            Function that solves for value of any given state using
            the minmax approach.
        """
        # self.depth is the maximum depth. here we just want to evaluate
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        # if agent is pacman, call the maximiser
        if agent == 0:
            result = self.maxFunc(gameState, depth, alpha, beta)
        # if agent is ghost, call the minimiser
        else:
            result = self.minFunc(gameState, agent, depth, alpha, beta)
        return result
            
    def minFunc(self, gameState, agent, depth, alpha, beta):
        """
          Function that returns the minimum value of the successorStates.
          The min function applies purely to the Ghosts
        """
        # initiate value of inf
        minActionValue = float("inf")
        # check the number of ghosts
        numGhosts = gameState.getNumAgents()-1
        # get legal actions for current active agent
        newActions = gameState.getLegalActions(agent)
        # for each action, get the new state
        for action in newActions:
            newState = gameState.generateSuccessor(agent, action)
            # if we are on the last ghost, go to pacman
            if agent == numGhosts:
                # call the maxfunction and increase the depth
                actionValue = self.value(newState, 0, depth+1, alpha, beta)
            # if not on the last agent, still need to check other ghost moves
            else:
                # action value will come from the minFunc of the next agent
                actionValue = self.value(newState, agent+1, depth, alpha, beta)        
            # if the action value is smaller, choose this value
            if actionValue < minActionValue:
                minActionValue = actionValue

            if actionValue < alpha:
                return actionValue

            beta = min(beta,actionValue)
                
        # if there were no new actions, simply return the current state value
        if newActions == []:
            minActionValue = self.evaluationFunction(gameState)

        # returns the minimum value calculated across actions
        return minActionValue

    def maxFunc(self, gameState, depth, alpha, beta):

        # initialise value of -inf
        # action Value should not be defined...
        maxActionValue = -float("inf")
        bestAction = ""
        # get legal actions to find new states
        newActions = gameState.getLegalActions(0)
        # for each possible action, get the states and their values
        for action in newActions:
            # for each action, calculate the sucessor state and value
            newState = gameState.generateSuccessor(0, action)
            # get value from minfunc, starting with ghost 1
            actionValue = self.value(newState, 1, depth, alpha, beta)
            # if the actionValue is greater than the previous max, best option
            if actionValue > maxActionValue:
                maxActionValue = actionValue
                bestAction = action
                if depth == 0:
                    self.action = bestAction
            if actionValue > beta:
                return actionValue
                
            alpha = max(alpha,actionValue)

        if newActions == []:
            maxActionValue = self.evaluationFunction(gameState)
            #alpha = max(alpha,actionValue)
        
        
        return maxActionValue

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # define initial depth as 1
        depth = 0
        self.action = []
        
        # want to find the sucessors for pacman (1-4 options) and
        # the successors for ghosts (1-4 options * number of ghosts)

        # start by initiating the max function for the current state
        bestAction = self.value(gameState, 0, depth)

        return self.action
        
        util.raiseNotDefined()

    def value(self, gameState, agent, depth):
        """
            Function that solves for value of any given state using
            the minmax approach.
        """
        if depth == self.depth:
            result = self.evaluationFunction(gameState)
            return result
        
        if agent == 0:
            result = self.maxFunc(gameState, depth)
            #print "max",result
        else:
            result = self.randFunc(gameState, agent, depth)
            #print "min",result
        
        return result
            
    def randFunc(self, gameState, agent, depth):
        """
          Function that returns the value of a random action.
          The random function applies purely to the Ghosts
        """
        # initiate value of inf
        minActionValue = float("inf")
        scores = []
        avgScore = 0
        # check the number of agents
        numAgents = gameState.getNumAgents()-1
        # get legal actions for current active agent
        newActions = gameState.getLegalActions(agent)
        if newActions == []:
            return self.evaluationFunction(gameState)
        # for each action, get the new state
        for action in newActions:
            newState = gameState.generateSuccessor(agent, action)
            # if we are on the last agent, get values
            if agent == numAgents:
                # call the maxfunction and increase the depth
                scores += [self.value(newState, 0, depth+1)]
            # if not on the last agent, still need to check other ghost moves
            else:
                # action value will come from the minFunc of the next agent
                scores += [self.value(newState, agent+1, depth)]
                # if the action value is smaller, choose this value
              
        # if there were no new actions, simply return the current state value
        for score in scores:
            avgScore += score
        avgScore /= len(scores)

        # returns the minimum value calculated across actions
        return avgScore

    def maxFunc(self, gameState, depth):

        # initialise value of -inf
        # action Value should not be defined...
        maxActionValue = -float("inf")
        bestAction = ""
        # get legal actions to find new states
        newActions = gameState.getLegalActions(0)
        # for each possible action, get the states and their values
        for action in newActions:
            # for each action, calculate the sucessor state and value
            newState = gameState.generateSuccessor(0, action)
            # get value from minfunc, starting with ghost 1
            actionValue = self.value(newState, 1, depth)
            # if the actionValue is greater than the previous max, best option
            if actionValue > maxActionValue:
                maxActionValue = actionValue
                bestAction = action
        if newActions == []:
            maxActionValue = self.evaluationFunction(gameState)

        self.action = bestAction
        
        return maxActionValue
        
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    pacmanLocation = currentGameState.getPacmanPosition()
    numFood = len(foodList)
    #print "sc",score,"nF",numFood

    # try to move away from ghosts

    # also distance to closes food
    # furthest food
    if numFood > 0:
        closestFood = float("inf")
        furthestFood = -float("inf")
        for food in foodList:
            closestFood = min(closestFood,util.manhattanDistance(food,pacmanLocation))
            furthestFood = max(furthestFood,util.manhattanDistance(food,pacmanLocation))

    else:
        closestFood = 0
        furthestFood = 0


    newGhostStates = currentGameState.getGhostStates()
    for ghostState in newGhostStates:
        ghostDistance = util.manhattanDistance(ghostState.getPosition(),pacmanLocation)        

    #if score < 0:
        #print "s",score,"nF",numFood,"cF",closestFood,"fF",furthestFood
    result = 2*score - numFood - closestFood - furthestFood + 0.5*ghostDistance
    
    return result
"""
# Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        evalGhostPos = 0.0
        testGhostPos = 0.0
        
        for state in newGhostStates:
            # get distance to ghosts
            # want this to get very bad, very close
            # approaches zero as it gets bad
            testGhostPos = abs(newPos[1] - state.getPosition()[1]) + abs(newPos[0] - state.getPosition()[0]) + 1
            #print state.getPosition(),"------",newPos,"-----",testGhostPos
            evalGhostPos += -100.0 / testGhostPos**3
            
        # evalGhostPos = evalGhostPos

        #if evalGhostPos > -2:
        #    evalGhostPos = 0

        evalFoodPos = 0
        evalFoodCount = 0
        evalFoodEat = 0
        minFoodDist = 1000
        
        for food in newFood.asList():
            # calculate the net distance from food.
            # Is it better to pick just a few?
            # add each distance
            # must divide by the total count of food to make sure changes <=1
            evalFoodPos += ((newPos[1] - food[1])**2 + (newPos[0] - food[0])**2)**(1.0/2.0)
            evalFoodCount += 1.0
            if (newPos[1] - food[1]) + (newPos[0] - food[0]) == 0:
                evalFoodEat = 1
            foodDist = abs(newPos[1] - food[1]) + abs(newPos[0] - food[0])
            if foodDist < minFoodDist:
                minFoodDist = foodDist
            
        # evalFoodPos should be strongest when there aren't many foods left
        evalFoodPos = -evalFoodPos/(evalFoodCount+1)
        evalFoodPos = 1.0/(minFoodDist+1)
        # pacman will thrash when he gets stuck between an even area
        # if values are even, always head left?
        #if evalFoodEat == 1:
        #    evalFoodPos = 0
        #import random
        #evalFoodPos = evalFoodPos + 2*random.random()
        # using evalFoodPos will just put Pacman in the middle of all the food always.
        # let's try going to the nearest food
        # if action == "West" or action == "South":
        #     evalFoodPos = evalFoodPos + 1
                
        # evalFoodCount = -100.0/evalFoodCount
        
        evalFood = -10000.0*(len(newFood.asList())+1)

        evalScore = -100.0/(currentGameState.getScore()+0.5)
        
        myEvalFun = evalFoodPos -evalFoodCount + evalGhostPos #+ evalFoodCount#+ evalFoodPos




    
    util.raiseNotDefined()
"""
# Abbreviation
better = betterEvaluationFunction

