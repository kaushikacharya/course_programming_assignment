# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #print "Start:", problem.getStartState()
    #print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    closed_state_list = set() # set of states already expanded
    # element of stack: [state,action path from start state]
    isGoalReached = False
    stack_obj = util.Stack()
    stack_obj.push( (problem.getStartState(),[]) )
    while not stack_obj.isEmpty():
        elem = stack_obj.pop()
        elem_state = elem[0]
        elem_action_history = elem[1]

        if problem.isGoalState(elem_state):
            isGoalReached = True
            break # reached goal

        if elem_state in closed_state_list:
            continue # already expanded
        else:
            closed_state_list.add(elem_state)

        for successor in problem.getSuccessors(elem_state):
            successor_state = successor[0]
            successor_action = successor[1]
            if successor_state in closed_state_list:
                continue
            # http://stackoverflow.com/questions/2612802/how-to-clone-a-list-in-python
            successor_action_history = list(elem_action_history)
            successor_action_history.append(successor_action)
            stack_obj.push((successor_state,successor_action_history))

    if not isGoalReached:
        elem_action_history = []
    return elem_action_history

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    closed_state_set = set()
    # element of queue: [state,action path from start state]
    isGoalReached = False
    queue_obj = util.Queue()
    queue_obj.push( (problem.getStartState(), []) )
    closed_state_set.add(problem.getStartState())
    while not queue_obj.isEmpty():
        elem = queue_obj.pop()
        elem_state = elem[0]
        elem_action_history = elem[1]

        if problem.isGoalState(elem_state):
            isGoalReached = True
            break

        for successor in problem.getSuccessors(elem_state):
            successor_state = successor[0]
            if successor_state in closed_state_set:
                continue
            successor_action_history = list(elem_action_history)
            successor_action_history.append(successor[1])
            closed_state_set.add(successor_state)
            queue_obj.push( (successor_state,successor_action_history) )

    if not isGoalReached:
        elem_action_history = []
    return elem_action_history
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    # http://stackoverflow.com/questions/12806452/whats-the-difference-between-uniform-cost-search-and-dijkstras-algorithm
    closed_state_set = set()
    # item of priority queue: [state, action path from start state, priority]
    # cost associated with item: total path cost
    isGoalReached = False
    pq_obj = util.PriorityQueue()
    pq_obj.push( (problem.getStartState(), [], 0), 0)
    
    while not pq_obj.isEmpty():
        elem = pq_obj.pop()
        elem_state = elem[0]
        elem_action_history = elem[1]
        elem_priority = elem[2]
        
        if problem.isGoalState(elem_state):
            isGoalReached = True
            break
            
        if elem_state in closed_state_set:
            continue # already expanded. Note that same state may get added to priority queue with different priorities.
        else:
            closed_state_set.add(elem_state)
            
        for successor in problem.getSuccessors(elem_state):
            successor_state = successor[0]
            successor_action = successor[1]
            successor_cost = successor[2] # cost of edge [elem_state,successor state]
            if successor_state in closed_state_set:
                continue
            successor_action_history = list(elem_action_history)
            successor_action_history.append(successor_action)
            successor_priority = elem_priority + successor_cost
            pq_obj.push( (successor_state, successor_action_history, successor_priority) , successor_priority)
            
    if not isGoalReached:
        elem_action_history = []
    return elem_action_history
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    closed_state_set = set()
    # item of priority queue: [state, action path from start state, path cost]
    # f(n) = g(n) + h(n); path cost = g(n), priority = f(n)
    isGoalReached = False
    pq_obj = util.PriorityQueue()
    heuristic_cost = heuristic(problem.getStartState(), problem)
    pq_obj.push( (problem.getStartState(), [], 0), 0+heuristic_cost)
    
    while not pq_obj.isEmpty():
        elem = pq_obj.pop()
        elem_state = elem[0]
        elem_action_history = elem[1]
        elem_path_cost = elem[2]
        
        if problem.isGoalState(elem_state):
            isGoalReached = True
            break
            
        if elem_state in closed_state_set:
            continue # already expanded. Note that same state may get added to priority queue with different priorities.
        else:
            closed_state_set.add(elem_state)
            
        for successor in problem.getSuccessors(elem_state):
            successor_state = successor[0]
            successor_action = successor[1]
            successor_cost = successor[2] # cost of edge [elem_state,successor state]
            if successor_state in closed_state_set:
                continue
            successor_action_history = list(elem_action_history)
            successor_action_history.append(successor_action)
            successor_path_cost = elem_path_cost + successor_cost
            successor_priority = successor_path_cost + heuristic(successor_state, problem)
            pq_obj.push( (successor_state, successor_action_history, successor_path_cost) , successor_priority)
    
    if not isGoalReached:
        elem_action_history = []
    return elem_action_history
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
