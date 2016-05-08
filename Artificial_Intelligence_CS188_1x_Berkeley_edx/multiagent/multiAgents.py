# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
        # https://courses.edx.org/courses/BerkeleyX/CS188.1x/2013_Spring/discussion/forum/undefined/threads/516170db3c253f120000008c
        # Thread topic: Pacman won't eat the last remaining food pellet
        foodPosList = newFood.asList()
        if len(foodPosList) > 0:
            list_food_dist = []
            for foodPos in foodPosList:
                dist = manhattanDistance(newPos,foodPos)
                dict_food = {}
                dict_food['pos'] = foodPos
                dict_food['distance'] = dist
                list_food_dist.append(dict_food)
            # now sort based on distance
            list_food_dist = sorted(list_food_dist, key=lambda elem:elem['distance'])
            
            # choose the nearest food which is closer to pacman compared to its distance from the ghosts.
            is_safe_food_found = False
            for food_index in range(0,len(list_food_dist)):
                foodPos = list_food_dist[food_index]['pos']
                is_food_closer_to_pacman = True
                for ghostState in newGhostStates:
                    # ignore the scared ghost
                    if ghostState.scaredTimer == 0:
                        ghostPos = ghostState.getPosition()
                        dist = manhattanDistance(foodPos,ghostPos)
                        if dist <= list_food_dist[food_index]['distance']:
                            is_food_closer_to_pacman = False
                            break
                if is_food_closer_to_pacman:
                    # current food is closer to pacman compared to the distance with the ghosts
                    successorGameState.data.score += 1.0/list_food_dist[food_index]['distance']
                    is_safe_food_found = True
                    break # no need to check rest of the foods which are at bigger distances from the pacman

            if is_safe_food_found == False:
                # assign score based on the distance from closest unscared ghost
                # TBD(done): if ghost(s) are beyond a certain distance from pacman, then pacman can procrastinate its worry about ghost.
                #      It can score according to the nearest food.
                min_dist = None
                for ghostState in newGhostStates:
                    # ignore the scared ghost
                    if ghostState.scaredTimer == 0:
                        ghostPos = ghostState.getPosition()
                        dist = manhattanDistance(newPos,ghostPos)
                        if min_dist == None:
                            min_dist = dist
                        elif dist < min_dist:
                            min_dist = dist

                if min_dist != None:
                    if min_dist < 4:
                        successorGameState.data.score += min_dist
                    else:
                        # unscared nearest ghost is quite far to worry now
                        food_index = 0
                        successorGameState.data.score += 1.0/list_food_dist[food_index]['distance']
                        # https://courses.edx.org/courses/BerkeleyX/CS188.1x/2013_Spring/discussion/forum/undefined/threads/514f795ef2d7671000000072
                        # "I solved this problem by having Pacman ignore the ghosts when they were further than a certain distance away."

        '''
        score = 0
        if len(foodPosList) > 0:
            # get the distance from Pacman to closest food
            min_dist = newFood.width + newFood.height
            for foodPos in foodPosList:
                dist = manhattanDistance(newPos,foodPos)
                if dist < min_dist:
                    min_dist = dist
            score = 1.0/min_dist
        
        # consider the ghost only when its not scared
        for ghost_index in range(0,len(newScaredTimes)):
            if newScaredTimes[ghost_index] == 0:
                ghostPos = newGhostStates[ghost_index].getPosition()
                dist = manhattanDistance(newPos,ghostPos)
                score += dist
        
        return score
        '''
        return successorGameState.getScore()

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
        numAgents = gameState.getNumAgents()
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = []
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0,action)
            scores.append(self.getValue(successorState,1,0))
            
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        
        '''
        for action in legalMoves:
            succPacmanState = gameState.generateSuccessor(0,action)
            curGhost1State = succPacmanState.getGhostState(1) # agent state not game state
            legalMovesGhost1State = succPacmanState.getLegalActions(1)
            for succAction in legalMovesGhost1State:
                succGhost1State = succPacmanState.generateSuccessor(1,succAction)
                gh = 0
        '''
        # util.raiseNotDefined()
        return legalMoves[chosenIndex]
        
    # --- Following functions are added by KA ---
    def getValue(self, gameState, agentIndex, depth=0):
        agentIndex = agentIndex % gameState.getNumAgents()
        if agentIndex == 0:
            # completed a depth
            depth += 1
            if depth == self.depth:
                # terminal state
                return gameState.data.score
                
        if agentIndex == 0:
            return self.getMaxValue(gameState,agentIndex,depth)
        else:
            return self.getMinValue(gameState,agentIndex,depth)
        
    def getMaxValue(self, gameState, agentIndex, depth):
        # used by pacman
        val = None
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex,action)
            if val == None:
                val = self.getValue(successorState, agentIndex+1, depth)
            else:
                val = max(val, self.getValue(successorState, agentIndex+1, depth))
        
        return val
        
    def getMinValue(self, gameState, agentIndex, depth):
        # used by ghosts
        val = None
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex,action)
            if val == None:
                val = self.getValue(successorState, agentIndex+1, depth)
            else:
                val = min(val, self.getValue(successorState, agentIndex+1, depth))
        
        return val
    # --- 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

