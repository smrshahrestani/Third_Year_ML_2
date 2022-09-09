# mdpAgents.py
# parsons/20-nov-2017
# 
# Author2: Seyed Mohammad Reza Shahrestani - 3/12/2021
# 
# Version 1
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import math
import game
import util


# This class is copied from "mapAgents.py" (week 5 solutions)
# 
# A class that creates a grid that can be used as a map
#
# The map itself is implemented as a nested list, and the interface
# allows it to be accessed by specifying x, y locations.
#
class Grid:

    # Constructor
    #
    # Note that it creates variables:
    #
    # grid:   an array that has one position for each element in the grid.
    # width:  the width of the grid
    # height: the height of the grid
    #
    # Grid elements are not restricted, so you can place whatever you
    # like at each location. You just have to be careful how you
    # handle the elements when you use them.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)

        self.grid = subgrid


    # Print the grid out.
    def display(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print (self.grid[i][j],)
            # A new line after each line of the grid
            print()
            # A line after the grid
        print()


    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value


    def getValue(self, x, y):
        return self.grid[y][x]


    # Return width and height to support functions that manipulate the
    # values stored in the grid.
    def getHeight(self):
        return self.height


    def getWidth(self):
        return self.width


# A class that creates an MDP Agent to calculate utilities for the map
# and pacman's locations 
# the utility updates each time an action is made 
# (when a food is eaten, or a ghost moves)
class MDPAgent(Agent):

    # The constructor. 
    # Initialising the reward values for each action
    def __init__(self):
        print ("Running init!")
        self.EMPTY_BLOCK_REWARD = 0  # The reward for the empty block
        self.FOOD_REWARD = 0
        self.CAPSULE_REWARD = 0
        self.GHOST_REWARD = 0
        self.AROUND_GHOST_REWARD = 0    # The reward for the ghost surrounding blocks
        self.EDIBLE_GHOST_REWARD = 0    # The reward for eating the ghost when its edible
        self.INTENDED = 0.8 # The probability of going in the intended direction
        self.NOT_INTENDED = 0.1 # The probability of going to either sides
        self.GAMMA = 0  # Discount factor
        self.PRECISION = 0.001  # Precision of calculated utility in every iteration
        self.rewards = []
        self.totalRunsSoFar = 0 
        self.winsSoFar = 0
        self.eatGhosts = False  # If True, it attempts to eat ghosts, otherwise it runs away from them
        self.useDistanceToGhost = False # It used the calculateDistance() to define if a ghost is edible
        self.outfile = open('moves.txt', 'w')


    # Check if the layout is small or medium
    # retruns True for the medium classic 
    # and False for the small classic
    def isMediumClassic(self):
        height =  self.map.getHeight()
        if height > 10: return True
        else: return False


    # Setting Initial values based in the map layout 
    def setInitialValues(self):
        # mediumClassic layout
        if self.isMediumClassic():
            print ("Layout: mediumClassic")

            self.EMPTY_BLOCK_REWARD = -0.05
            self.FOOD_REWARD = 5
            self.CAPSULE_REWARD = 6
            self.GHOST_REWARD = -500
            self.AROUND_GHOST_REWARD = -200
            self.EDIBLE_GHOST_REWARD = 100
            self.INTENDED = 0.8
            self.NOT_INTENDED = 0.1
            self.GAMMA = 0.7
            self.PRECISION = 0.000001

        # samllClassic layout
        else:
            print ("Layout: smallClassic")

            self.EMPTY_BLOCK_REWARD = -2
            self.FOOD_REWARD = 30
            self.CAPSULE_REWARD = 20
            self.GHOST_REWARD = -1200
            self.AROUND_GHOST_REWARD = -400
            self.EDIBLE_GHOST_REWARD = 0
            self.INTENDED = 0.8
            self.NOT_INTENDED = 0.1
            self.GAMMA = 0.7
            self.PRECISION = 0.001
        
        print ("")


    # This function will be called when the pacman hasn't won enough games
    # This function will re assing rewards to each object in the game 
    def setRewardsWithCaution(self):
        # mediumClassic layout
        if self.isMediumClassic():
            self.EMPTY_BLOCK_REWARD = -0.1
            self.FOOD_REWARD = 5
            self.CAPSULE_REWARD = 5
            self.GHOST_REWARD = -500
            self.AROUND_GHOST_REWARD = -300
            self.EDIBLE_GHOST_REWARD = 5
            self.GAMMA = 0.7
            self.PRECISION = 0.000001

        # samllClassic layout
        else:
            self.EMPTY_BLOCK_REWARD = -1
            self.FOOD_REWARD = 10
            self.CAPSULE_REWARD = 10
            self.GHOST_REWARD = -400
            self.AROUND_GHOST_REWARD = -200
            self.EDIBLE_GHOST_REWARD = 1
            self.GAMMA = 0.7
            self.PRECISION = 0.001


    # This function is copied from "mapAgents.py" (week 5 solutions)
    # 
    # This function is run when the agent is created, and it has access
    # to state information, so we use it to build a map for the agent.
    def registerInitialState(self, state):
        print ("Running registerInitialState!")

        # Make a map of the right size
        self.makeMap(state)
        self.addWallsToMap(state)

        # Setting the initial rewards
        self.setInitialReward(state)

        # Initialise the reward values
        self.setInitialValues()

        # Update rewards
        self.updateRewards(state)


    # This is what gets run when the game ends.
    # In the case that pacman is not wining enough games, 
    # this function will decide what behaviours of pacman has to be changed
    def final(self, state):
        self.totalRunsSoFar += 1

        if state.isWin():
            self.winsSoFar += 1
            print ("I WON, YEAY! :)))")
        else: print ("Looks like I just died! :(((")

        # mediumClassic
        if self.isMediumClassic():
            if self.totalRunsSoFar > 16 and self.winsSoFar < 6:
                # self.eatGhosts = False
                self.setRewardsWithCaution()
                print ("-" * 40)
                print ("Pacman is using the new rewards!")

        else:
            if self.totalRunsSoFar > 14 and self.winsSoFar < 9:
                # self.eatGhosts = False
                self.setRewardsWithCaution()
                print ("-" * 40)
                print ("Pacman is using the new rewards!")

        print ("-" * 40)


    # This function is copied from "mapAgents.py" (week 5 solutions)
    # 
    # Make a map by creating a grid of the right size
    def makeMap(self, state):
        corners = api.corners(state)
        height = self.getLayoutHeight(corners)
        width = self.getLayoutWidth(corners)
        self.map = Grid(width, height)


    # This function is copied from "mapAgents.py" (week 5 solutions)
    # 
    # Functions to get the height and the width of the grid.
    #
    # We add one to the value returned by corners to switch from the
    # index (returned by corners) to the size of the grid (that damn
    # "start counting at zero" thing again).
    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1


    # This function is copied from "mapAgents.py" (week 5 solutions)
    # 
    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1


    # This function is copied from "mapAgents.py" (week 5 solutions)
    # 
    # Functions to manipulate the map.
    #
    # Put every element in the list of wall elements into the map
    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '%')
        

    #  Calculate the distance of the pacman to the ghost
    #  if the ghost is accessable to pacman withing the time limit of 
    #  being scared, the possitions and the scared time limit of ghosts will be returned
    #  with a boolean variable indicating if the pacman should attemp to eat the ghost or not.
    def calculateDistance(self,pacman, ghostsWithTime):
        pacmanDistanceToGhosts = []

        for i in ghostsWithTime:
            ghost_X = i[0][0]
            ghost_Y = i[0][1]
            pacman_X = pacman[0]
            pacman_Y = pacman[1]
            ghost_time = i[1]
            attempt = False

            distance = abs(ghost_X - pacman_X) + abs(ghost_Y - pacman_Y) 

            if ghost_time > distance*3/2: 
                attempt = True
            final = [(ghost_X, ghost_Y), ghost_time , distance, attempt]

            pacmanDistanceToGhosts.append(final)
            
        return pacmanDistanceToGhosts


    # setting initial reward
    # foods, capsules, empty space
    def setInitialReward(self,state):
        width = self.map.getWidth()
        height = self.map.getHeight()
        food = api.food(state)
        capsules = api.capsules(state)

        # Create the reward grid and initialise the values
        # Initialise EMPTY_BLOCK_REWARD to all the blocks (if they're not wall)
        self.rewards = Grid(width, height)
        for i in range(width):
            for j in range(height):
                if self.map.getValue(i, j) != "%":
                    self.rewards.setValue(i, j, self.EMPTY_BLOCK_REWARD)

        # Set the food reward
        for (food_X, food_Y) in food:
            self.rewards.setValue(food_X, food_Y, self.FOOD_REWARD +
            self.rewards.getValue(food_X, food_Y))

        # Set the capsule reward
        for (capsule_X, capsule_Y) in capsules:
            self.rewards.setValue(capsule_X, capsule_Y, self.CAPSULE_REWARD + self.rewards.getValue(food_X, food_Y))


    # Updated the reward value of each position
    def updateRewards(self, state):    
        ghosts = api.ghostStates(state)
        ghostsWithTime = api.ghostStatesWithTimes(state)
        self.setInitialReward(state)

        # Setting the rewards for blocks, ghosts and the block around ghosts
        for i in range(len(ghosts)):
            ghost_X = ghosts[i][0][0]
            ghost_Y = ghosts[i][0][1]
            s = ghosts[i][1]
            ghost_X = int(ghost_X)
            ghost_Y = int(ghost_Y)
            pacman = api.whereAmI(state)
            distance = self.calculateDistance(pacman,ghostsWithTime)
 
            # It uses calculateDistance() function to decide if the pacman can eat the ghost
            if self.useDistanceToGhost and distance[i][3]: 
                self.rewards.setValue(int(distance[i][0][0]),int(distance[i][0][1]),
                self.EDIBLE_GHOST_REWARD + 
                self.rewards.getValue(int(distance[i][0][0]),int(distance[i][0][1])))
            
            # If the ghost is still edible, the pacman will try to eat it
            elif not self.useDistanceToGhost and self.eatGhosts and ghostsWithTime[i][1] > 1:
                self.rewards.setValue(int(ghostsWithTime[i][0][0]),int(ghostsWithTime[i][0][1]), 
                self.EDIBLE_GHOST_REWARD + 
                self.rewards.getValue(int(ghostsWithTime[i][0][0]),int(ghostsWithTime[i][0][1])))

            #  If the ghosts are not edible
            else:
                # Set the ghost reward
                self.rewards.setValue(ghost_X, ghost_Y, self.GHOST_REWARD +
                self.rewards.getValue(ghost_X, ghost_Y))

                if self.AROUND_GHOST_REWARD != 0:
                    # Set the surrounding possitions of the ghosts
                    surroundingGhost = [(ghost_X - 1, ghost_Y), (ghost_X + 1, ghost_Y), (ghost_X, ghost_Y - 1), (ghost_X, ghost_Y + 1)]

                    # Filtering the walls around the ghost
                    noWallsAroundGhost = self.noWallsAround(state,surroundingGhost)
                    
                    # Set value for the surrounding blocks around the ghost
                    for (x, y) in noWallsAroundGhost:
                        self.rewards.setValue(int(x), int(y),
                                self.AROUND_GHOST_REWARD + 
                                self.rewards.getValue(int(ghost_X), int(ghost_Y)))            
                        

    # Creates the utility map
    # which maps each location to a reward value
    def utilityMap(self):
        width = self.map.getWidth()
        height = self.map.getHeight()

        utilGrid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                if self.map.getValue(i, j) == "%": 
                    utilGrid.setValue(i, j, "-")
                else: utilGrid.setValue(i, j, 0)
        return utilGrid


    # Translates directions to coordinates
    def coordinatesTraslate(self, dir):
        if (dir == 'North'): return (0, 1)
        elif (dir == 'West'): return (-1, 0)
        elif (dir == 'South'): return (0, -1)
        elif (dir == 'East'): return (1, 0)
        else: return (0, 0) 

   
    # Returns all 4 directions of a position
    def getAllDirections(self, current):
        x = current[0]
        y = current[1]
        return [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]


    # Returns a list of the intended positions and not intended sides possitions
    def getSurroundings(self, current, intended):
        current_X = current[0]
        current_Y = current[1]
        intended_X = intended[0]
        intended_Y = intended[1]

        if intended_X != current_X:
            return [(current_X, current_Y + 1), (intended_X, intended_Y), (current_X, current_Y - 1)]
        elif intended_Y != current_Y:
            return [(current_X - 1, current_Y), (intended_X, intended_Y), (current_X + 1, current_Y)]


    # Filtering the walls around the ghost
    def noWallsAround(self, state, surroundingGhost):
        return filter(lambda x: x not in api.walls(state), surroundingGhost)
 

    # Returns TRUE if the position is a wall
    def isWall(self, utilGrid, position):
        cond1 = position[0] < 0
        cond2 = position[1] < 0
        cond3 = position[0] >= len(utilGrid)
        cond4 = position[1] >= len(utilGrid[0])
        cond5 = utilGrid[position[0]][position[1]]
    
        return (cond1 or cond2 or cond3 or cond4 or cond5 == "-")


    # Calculates the utility of the nonDeterministic actions
    def calculateUtility(self, copyGrid, intendedAndSurroundings):
        intended_X = intendedAndSurroundings[1][0]
        intended_Y = intendedAndSurroundings[1][1]

        left_X = intendedAndSurroundings[0][0]
        left_Y = intendedAndSurroundings[0][1]

        right_X = intendedAndSurroundings[2][0]
        right_Y = intendedAndSurroundings[2][1]

        intended = self.INTENDED * copyGrid[intended_X][intended_Y]
        not_intended = self.NOT_INTENDED * (copyGrid[left_X][left_Y] + copyGrid[right_X][right_Y])

        return intended + not_intended


    # Gets all the possible directions of current position, intended
    # and not intended directions
    # Tt checks the walls, so if the pacman hits a wall stayes in the same place
    # Calculates utility of all the intended directions
    # Returns a list of maximised utilities
    def maxUtility(self, utilGrid, current):
        allDirections = self.getAllDirections(current)
        intendedAndSurroundings = []
        noWallIntendedAndSurroundings = []
        calculatedUtility = []

        # A list of all the all the intended and surrounding directions
        for intended in allDirections:
            intendedAndSurroundings.append(self.getSurroundings(current,intended))

        # Checks if the intended or not intended directions are heading to wall,
        # then it will replace them by the current location of the pacman
        for intended in intendedAndSurroundings:
            temp=[]
            for coordinates in intended:
                if self.isWall(utilGrid, coordinates):
                    temp.append(current)
                else:
                    temp.append(coordinates)
            noWallIntendedAndSurroundings.append(temp)

        # Calculates the utilities and returns a list
        for intendedAndSurroundings in noWallIntendedAndSurroundings:
            calculatedUtility.append(self.calculateUtility(utilGrid, intendedAndSurroundings))

        return max(calculatedUtility)


    # Calculates the final utility using the Bellman equation for all possitions in map
    # Keeps iterating until the values stop changing 
    def valueIteration(self, utilGrid):
        width = utilGrid.getWidth()
        height = utilGrid.getHeight()
        breakTheLoop = False

        while not breakTheLoop:
            copyGrid = []

            # Copies the grid
            for i in range(width):
                temp = []
                for j in range(height):
                    temp.append(utilGrid.getValue(i, j))
                copyGrid.append(temp)
            
            # Updates the utility
            for i in range(width):
                for j in range(height):
                    if not utilGrid.getValue(i, j) == "-":
                        utility = self.rewards.getValue(i, j) + self.GAMMA * self.maxUtility(copyGrid, (i, j))
                        utilGrid.setValue(i, j, utility)

            # Checks if the iteration has stabilised with a certain PRECISION
            for i in range(width):
                for j in range(height):
                    if utilGrid.getValue(i, j) != "-" and copyGrid[i][j] != "-":
                        if abs(utilGrid.getValue(i, j) - copyGrid[i][j]) > self.PRECISION: 
                            breakTheLoop = False
                        else: 
                            breakTheLoop = True

        return utilGrid


    # Returns the best direction for the pacman to make using the utility grid
    def getBestDirection(self, state, utilGrid):
        legal = self.getLegal(state)
        best_direction = Directions.STOP
        best_utility = -10000

        current_X = api.whereAmI(state)[0]
        current_Y = api.whereAmI(state)[1]

        for direction in legal:
            intended_X = self.coordinatesTraslate(direction)[0]
            intended_Y = self.coordinatesTraslate(direction)[1]

            new_X = current_X + intended_X
            new_Y = current_Y + intended_Y

            if utilGrid.getValue(new_X, new_Y) > best_utility:
                best_direction = direction
                best_utility = utilGrid.getValue(new_X, new_Y)

        return best_direction


    # Return all the legal actions excluding STOP action
    def getLegal(self, state):
        legal = api.legalActions(state)

        # removing the STOP actions in the medium classic
        if self.isMediumClassic():
            if Directions.STOP in legal:
                legal.remove(Directions.STOP)

        return legal
 

    # Updates the reward grid in every step
    # Creates the utility grid
    # Makes the action
    def getAction(self, state):

        legal = self.getLegal(state)
        self.updateRewards(state)
        utilGrid = self.valueIteration(self.utilityMap())
        best_direction = self.getBestDirection(state,utilGrid)


        move = best_direction

        # Log feature set and choice of move
        if move != Directions.STOP:
            self.outfile.write(api.getFeaturesAsString(state))
            if move == Directions.NORTH:
                self.outfile.write("0\n")
            elif move == Directions.EAST:
                self.outfile.write("1\n")
            elif move == Directions.SOUTH:
                self.outfile.write("2\n")
            elif move == Directions.WEST:
                self.outfile.write("3\n")
    

        return api.makeMove(best_direction, legal)



# THANK YOU FOR READING MY CODE ;)
