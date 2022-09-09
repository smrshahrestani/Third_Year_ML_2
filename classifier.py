# classifier.py
# Lin Li/26-dec-2021
# Author: Seyed Mohammad Reza Shahrestani
# Date: 22/02/2022
#
# Use the skeleton below for the classifier and insert your code here.


# data:
# [Wall-Up, Wall-Right, Wall-Down, Wall-Left, 
# Food-Up, Food-Right, Food-Down, Food-Left
# G1-Up_Left, G1-Up_Middle, G1-Up_Right,
# G1-Middle_left, G1-Middle_Right,
# G1-Down_Left, G1-Down_Middle, G1-Down_Right,
# G2-Up_Left, G2-Up_Middle, G2-Up_Right,
# G2-Middle_left, G2-Middle_Right,
# G2-Down_Left, G2-Down_Middle, G2-Down_Right, 
# Ghost-in-front-of-pacman]
# [North or East or South or West]


# I have chosen to use the Decision Tree Learning algorithm to decide which action
# should the Pacman take in every step
# I also used my previous code from AIN CW, called MDPAgent to generate more test data.
# Using the MDPAgent I ran the game for 1000 rounds (with a win rate of 94%) and I did manage to
# get around 133,000 new test data.
# Then I used the test data to train my decision tree, and this is the result of the decision tree:
# https://bit.ly/ML1-CW


from dis import dis
from logging import raiseExceptions
import math
from multiprocessing.sharedctypes import Array
from operator import le
from pickle import TRUE
from re import I
from turtle import left, right
from urllib.parse import _NetlocResultMixinBase
import numpy as np # for calculating log2 and creating matrix


# Remove a custom move from the data
# for testing purpose
def removeCustomMove(data, target, move):

    newData = []
    newTarget = []
    for i in range(len(data)):
        for j in range(len(target)):
            if target[j] != move:
                newData.append(data[i])
                newTarget.append(target[j])
    return newData, newTarget


# Removes the duplicates
def removeDuplicates(data):

     uniqueData = np.unique(np.array(data), axis=0)

     return uniqueData


# returns the sum of all the instances of classes
# [number of UPs, number of RIGHTs, number of DOWNs, number of LEFTs]
# 
def sumOfEachMove(target):

    allTargets = [0,0,0,0]

    for i in range(len(target)):
        if (target[i] == 0): allTargets[0] +=1 
        elif (target[i] == 1): allTargets[1] +=1 
        elif (target[i] == 2): allTargets[2] +=1 
        elif (target[i] == 3): allTargets[3] +=1    

    return allTargets


# Calculating the Entropy 
def calcEntropy(target):

    numberOfClasses = 4  # up - right - down - left
    sumOfMoves = sumOfEachMove(target)
    total = sum(sumOfMoves)
    entropy = 0

    for i in range(numberOfClasses):
        pi =  sumOfMoves[i] / total
        if (pi != 0): # to avoid devision by 0
            x = (pi * math.log2(pi)) 
            entropy -= x

    return entropy


# Calculating the Gain
# Formula: Gain(S,A) = H(S) − ∑ |Si|/|S| H(Si)
# in Si, i = 2 as we have 2 choice for each class
# S1 = when S is 1
# S0 = when S is 0
# 
def calcGain(data, target):

    # data = removeDuplicates(data)

    numberOfClasses = 4  # up - right - down - left
    numberOfFeature = len(data[0]) #25

    # Creating 2 matrix for storing entropy matrix
    entropyMatrix1 = np.zeros((numberOfClasses, numberOfFeature))
    entropyMatrix0 = np.zeros((numberOfClasses, numberOfFeature))

    total = sum(sumOfEachMove(target))
    entropy = calcEntropy(target)

    #  creating the entropy matrix 
    for i in range(len(data)):
        for j in range(numberOfFeature):
            if data[i][j] == 1 :
                classResult = (target[i])
                entropyMatrix1[classResult][j] +=1
            else:
                classResult = (target[i])
                entropyMatrix0[classResult][j] +=1

    entropyArray = []

    for i in range(numberOfFeature):
        numeratorS0= 0
        numeratorS1 = 0
        x0, x1 = 0, 0
        s0,s1 = 0, 0
        p0, p1 = 0,0

        for j in range(numberOfClasses):
            numeratorS0 += entropyMatrix0[j][i]
            numeratorS1 += entropyMatrix1[j][i]

        for k in range(numberOfClasses):

            if numeratorS0 != 0: # to avoid devision by 0
                p0 = entropyMatrix0[k][i] / numeratorS0

            if p0 != 0: # to avoid devision by 0
                x0 += (-1) * p0 * math.log2(p0)

            if numeratorS1 != 0: # to avoid devision by 0
                p1 = entropyMatrix1[k][i] / numeratorS1

            if p1 != 0: # to avoid devision by 0
                x1 += (-1) * p1 * math.log2(p1)

        s0 = (numeratorS0 / total) * x0
        s1 = (numeratorS1 / total) * x1

        entropyArray.append(entropy - (s0 + s1))

    return entropyArray


# Returns the most used target in the parent class
def pluralityValue(target):

    sumList = sumOfEachMove(target)
    bestMove = (sumList.index(max(sumList)))
    
    return bestMove


# This is a recursive function that uses the data and target arrays
# to learn the best move
# it stands for Decision Tree Learning
# it consists of 4 different parts
# 1: if we do not have enough examples, it uses the plurality algorithm to decide what to do
# 2: if all the examples have the same classification, then returns the classification
# 3: if we have data, but it does not fit any class, it uses the plurality algorithm
# 4: if non of the above: finds the best attribute based on the Gain function and 
# creates a tree with the attribute being the root.
# then it splits the data based on the best attribute into 2 groups of 0s and 1s
# and then creates 2 subtrees and return the split data to the subtrees
# in the way that the left subtree has the 0 values and the right subtree has the 1 values
# 
def DTL(data, target, attributes, parentData, parentTarget):

    # data = removeDuplicates(data)

    remain = []
    isSame= True

    # Cheking if all the classifications are the same
    for i in range(len(data)-1):
        isSame = target[i+1] == target[i] and isSame

    if len(data) == 0: return BinaryDecisionTree(pluralityValue(parentTarget), None, None, None) # 1
    elif isSame: return BinaryDecisionTree(target[0], None, None, None) # 2
    elif len(attributes) == 0: return BinaryDecisionTree(pluralityValue(target), None, None, None) # 3
    else: # 4

        gains = calcGain(data, target)
        validGains = [(attr,gain) for (attr,gain) in enumerate(gains) if attr in attributes] 
        # print(calcGain(data, target))
        best = max(validGains, key=lambda x: x[1])[0]
        remain = attributes # remain = remove best from attribute
        remain.remove(best)


        leftData = []
        rightData = []
        leftTarget = []
        rightTarget = []

        #  create a left and right subtree
        # left-> where best is 0 
        # right -> where best is 1 

        for i in range(len(data)):  # split data and target based on best index
            if data[i][best] == 1:
                rightData.append(data[i])
                rightTarget.append(target[i])
            else:
                leftData.append(data[i])
                leftTarget.append(target[i])

        rightTree= DTL(rightData, rightTarget, remain, data, target)
        leftTree= DTL(leftData, leftTarget, remain, data, target)

        return BinaryDecisionTree(None, leftTree, rightTree, best)


# A Decision Tree 
# This class handles the tree classification
# an instance of this class can either be a subtree or a value
# A subtree consists of a left and right node which can also be
# a subtree or a value
# 
class BinaryDecisionTree:

    isNode = False

    def __init__(self, value, nodeLeft, nodeRight, attribute):

        self.attribute = attribute
        if value is not None: 
            self.left = None
            self.right = None
            self.value = value

        else:
            self.value = None
            if nodeLeft is not None: 
                self.left = nodeLeft
            if nodeRight is not None: 
                self.right = nodeRight

    # A function to print the decision tree
    # it shows the root on the left, and it shows the 
    # left node by: 'root' -->|0| 'something'
    # and the right node by: 'root' -->|1| 'something'
    # where 'something' can be another tree or it can be a value
    # it shows the values with a 'v' behind it
    # for example 6 -->|0| v1 this indicates that root 6 has a left value of 1
    # And 5 -->|1| 1 indicates that root 5 has a right value which is also a tree and its root number 1.
    # You can copy the result and paste it into this website under 'graph TD' to see the visual graph:
    # https://bit.ly/ML1-Graph 
    # 
    def show(self):
        if self.value is not None:
            return ''
        final = str(self.attribute) + " -->|0| " + (str(self.left.attribute) if self.left.value is None else ("v"+str(self.left.value))) + "\n"
        final += str(self.attribute) + " -->|1| " + (str(self.right.attribute) if self.right.value is None else ("v"+str(self.right.value))) + "\n"
        final += self.left.show()
        final +=  self.right.show()
        
        return final

        



class Classifier:

    # only runs for the initialisation
    def __init__(self):

        print("Classifier Init Function")
        self.tree = None  # Initialising the tree
        pass
    
    # it runs when the Pacman has either won or lost. (when the game finishes)
    def reset(self):

        print("Classifier Reset Function")
        pass
    
    # runs once to train the data
    def fit(self, data, target):

        print("Classifier Fit Function")

        lst= [i for i in range(25)] # Create a list [0,1,2,...,24]
        self.tree = DTL(data, target, lst, [], []) # Start the learning process and storing the result
        print(self.tree.show()) # Prints the tree

        pass

    # runs every step of the game
    def predict(self, data, legal=None):

        currentNode = self.tree

        # While the tree has children it goes through all the children
        # and predicts what to do based on the decision tree
        while (currentNode.left is not None or currentNode.right is not None):

            if (data[currentNode.attribute] == 0):
                currentNode = currentNode.left
            elif (data[currentNode.attribute] == 1):
                currentNode = currentNode.right
        
        print(currentNode.value)
        
        return currentNode.value
        


# THANK YOU FOR READING MY CODE ;)