# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        values2 = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Vk+1 = max_a ( sum_s' T * (R + Df * Vk(s')))
        # may need to separate into values old and values new        
        iteration = 1
        self.directions = util.Counter()
        # iterate up to a certain amount
        while iterations >= 0:
            # look at each possible state in the mdp
            for state in mdp.getStates():
                # initialise the maxQValue and then look at all possible actions
                maxQValue = -float("inf")
                for action in mdp.getPossibleActions(state):
                    # Calculate QValue
                    # calculated QValues are wrong for iteration 0!!!!
                    QValue = self.computeQValueFromValues(state,action)
                    #print state,QValue
                    if QValue > maxQValue:
                        maxQValue = QValue
                        self.directions[state] = action
                if mdp.getPossibleActions(state) == ():
                    maxQValue = self.values[state]
                    # what should the value be in this case?
                values2[state] = maxQValue
                #print values2
            self.values = values2
            iterations -= 1
        iterations = self.iterations
        return
            

        
        
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # block this to sucessfully run without error
        return self.values[state]
        util.raiseNotDefined()

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        sumState = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
        #print "state",state
        #print "action", action
        #print "trans",transitions
       
        for trans in transitions:
            transProb = trans[1]
            transState = trans[0]
            reward = self.mdp.getReward(state,action,transState)
            #print "R", reward
            # the magic equation
            sumState += transProb * (reward + self.discount * self.getValue(transState))
        return sumState
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        # computes the best action for each state
        if state == "TERMINAL_STATE":
            return None
        return self.directions[state]
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        "Returns the Q value from function"
        return self.computeQValueFromValues(state, action)
