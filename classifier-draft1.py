# classifier.py
# Lin Li/26-dec-2021
# Author: Seyed Mohammad Reza Shahrestani
# Date: 07/02/2022
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

# target
# [where to go]

from dis import dis
from logging import raiseExceptions
import math
from multiprocessing.sharedctypes import Array
from operator import le
from pickle import TRUE
from re import I
from turtle import left, right
from urllib.parse import _NetlocResultMixinBase
import numpy as np #Added by me!

# useless.
def euclideanDistance(a,b):
    x1 , y1 = a[0], a[1]
    x2 , y2 = b[0], b[1]
    distance = math.sqrt(((abs(x2 - x1) + abs(y2 - y1))**2))
    print (distance)
# euclideanDistance([2,3], [10,3])


# useless
def calcProbabiliy(a,b):
    if (a == "0"):
        print ("Wall")
    elif (a == "1"):
        print ("Food")
    elif (a == "2"):
        print ("g1")
    elif (a == "3"):
        print ("g2")
    elif (a == "4"):
        print ("gFront")
    else:
        print ("ERROR")


# returns the sum of all all the instances of classes
# [number of UPs, number of RIGHTs, number of DOWNs, number of LEFTs]
def sumOfEachMove(target):
    allTargets = [0,0,0,0]
    for i in range(len(target)):
        if (target[i] == 0): allTargets[0] +=1 
        elif (target[i] == 1): allTargets[1] +=1 
        elif (target[i] == 2): allTargets[2] +=1 
        elif (target[i] == 3): allTargets[3] +=1    
    return allTargets


# associate data to its class
def getClassifidData(data , target):

    classifiedData = np.zeros((4, len(data)), dtype=int)

    for i in range(len(data)):
        if target[i] == 0:
            classifiedData[0, i] += 1
        if target[i] == 1:
            classifiedData[1, i] += 1
        if target[i] == 2:
            classifiedData[2, i] += 1
        if target[i] == 3:
            classifiedData[3, i] += 1
        
    return classifiedData

# # store probability of each data with its class
def getProbabilityArray(data, target):

    probabilityArray = np.zeros((4, len(data)), dtype=float)
    sumArray = sumOfEachMove(target)
    classifiedData = getClassifidData(data, target)

    for i in range(len(sumArray)):
        for j in range(len(data)):
            probabilityArray[i,j] = classifiedData[i,j] / float(sumArray[i])
    print(probabilityArray)



# formula: find the probability
# p(a) = p/p+n(total)
# 
def calcEntropy(data, target):


    numberOfClasses = 4  # up - right - down - left
    numberOfFeature = len(data[0]) #25

    sumOfMoves = sumOfEachMove(target)
    total = sum(sumOfMoves)
    entropy = 0
    for i in range(numberOfClasses):
        pi =  sumOfMoves[i] / total
        # print(pi)
        if (pi != 0):
            x = (pi * math.log2(pi)) 
            entropy -= x

    return entropy


def calcGain(data, target):

    numberOfClasses = 4  # up - right - down - left
    numberOfFeature = len(data[0]) #25

    total = sum(sumOfEachMove(target))

    entropyMatrix1 = np.zeros((numberOfClasses, numberOfFeature))
    entropyMatrix0 = np.zeros((numberOfClasses, numberOfFeature))
    entropy = calcEntropy(data, target)

    #  creating the entropy matrix 
    for i in range(len(data)):
        for j in range(numberOfFeature):
            if data[i][j] == 1 :
                classResult = (target[i])
                entropyMatrix1[classResult][j] +=1
            else:
                classResult = (target[i])
                entropyMatrix0[classResult][j] +=1



    #
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


    # print(entropyArray)
    return entropyArray


def pluralityValue(data,target):

    sumList = sumOfEachMove(target)
    bestMove = (sumList.index(max(sumList)))

    # calculate parents node probability distribution
    
    return bestMove




def DTL(data, target, attributes, parentData, parentTarget):

    # print("data: ", data)
    # print("target: ", target)
    # print("attr: ", attributes)
    # print("parentData: ", parentData)
    # print("parentTarget: ", parentTarget)

    # data = examples[:-1]
    # target = examples[-1]
    remain = []
    isSame= True
    for i in range(len(data)-1):
        isSame = target[i+1] == target[i] and isSame

    # print("attr: ", attributes)

    if len(data) == 0: return BinaryDecisionTree(pluralityValue(parentData, parentTarget), None, None, None) 
    elif isSame: return BinaryDecisionTree(target[0], None, None, None) 
    elif len(attributes) == 0: return BinaryDecisionTree(pluralityValue(data, target), None, None, None)
    else:
        gains = calcGain(data, target)
        # [0.45,0.24,...]
        validGains = [(attr,gain) for (attr,gain) in enumerate(gains) if attr in attributes]
        best = max(validGains, key=lambda x: x[1])[0]
        
        # print("valid Gains: ", validGains)

        print("this is the best: ",best)
        # print("tpe of the attributes: ", type(attributes))
        remain = attributes # remain = remove best from attribute
        remain.remove(best)
        # print("remain: ", remain)

        leftData = []
        rightData = []
        leftTarget = []
        rightTarget = []

        #  create a left and right subtree
        # left-> where best is 0 
        # right -> where best is 1 

        for i in range(len(data)): # split data and target based on best index
            if data[i][best] == 1:
                rightData.append(data[i])
                rightTarget.append(target[i])
            else:
                leftData.append(data[i])
                leftTarget.append(target[i])

        rightTree= DTL(rightData, rightTarget, remain, data, target)
        leftTree= DTL(leftData, leftTarget, remain, data, target)

        return BinaryDecisionTree(None, leftTree, rightTree, best) #best??



# def DTL2(data, target, attributes, parentData, parentTarget):

#     # data = examples[:-1]
#     # target = examples[-1]
#     remain = []
#     isSame= True
#     for i in range(len(data)-1):
#         isSame = target[i+1] == target[i] and isSame

    print("attr: ", attributes)
#     if len(data) == 0: return BinaryDecisionTree(pluralityValue(parentData, parentTarget), None, None, None) 
#     elif isSame: return BinaryDecisionTree(target[0], None, None, None) 
#     elif len(attributes) == 0: return BinaryDecisionTree(pluralityValue(data, target), None, None, None)
#     else:

#         gains = calcGain(data, target)
#         best = gains.index(max(gains))

#         myTree = BinaryDecisionTree()

#         # print("this is the best: ",best)
#         # print("tpe of the attributes: ", type(attributes))
#         remain = attributes # remain = remove best from attribute
#         remain.pop(best)
#         # print("remain: ", remain)

#         leftData = []
#         rightData = []
#         leftTarget = []
#         rightTarget = []

#         #  create a left and right subtree
#         # left-> where best is 0 
#         # right -> where best is 1 

#         for i in range(len(data)): # split data and target based on best index
#             if data[i][best] == 1:
#                 rightData.append(data[i])
#                 rightTarget.append(target[i])
#             else:
#                 leftData.append(data[i])
#                 leftTarget.append(target[i])

#         rightTree= DTL(rightData, rightTarget, remain, data, target)
#         leftTree= DTL(leftData, leftTarget, remain, data, target)

#         return BinaryDecisionTree(None, leftTree, rightTree, best) #best??



# class BinaryDecisionTree2:
#     def __init__(self,root, left, right, value):


class BinaryDecisionTree:
    isNode = False

    def __init__(self, value, nodeLeft, nodeRight, attribute):
        # print("this is the attribute ", attribute)
        # print("this is the value ", value)
        self.attribute = attribute
        if value is not None: 
            self.left = None
            self.right = None
            self.value = value

        else:
            self.value = None
            if nodeLeft is not None: #changed elif to if
                self.left = nodeLeft
                # print("left node: ", nodeLeft)
            if nodeRight is not None: #changed elif to if
                self.right = nodeRight
                # print("right node: ", nodeRight)
    
    def show(self):
        if self.value is not None:
            return ''
        final = str(self.attribute) + " -->|0| " + (str(self.left.attribute) if self.left.value is None else ("v"+str(self.left.value))) + "\n"
        final += str(self.attribute) + " -->|1| " + (str(self.right.attribute) if self.right.value is None else ("v"+str(self.right.value))) + "\n"
        final += self.left.show()
        final +=  self.right.show()
        
        return final

        # print("value: ", self.value)
        # print("left: ", self.left)
        # print("right:  ", self.right)


    
    # def left(self):
    #     pass

    # def right(self):
    #     pass

    # def hasChilderen(self):
    #     hasChilderen = False

    #     # 
         
    #     return hasChilderen



        



class Classifier:
    # only runs for the initialisation
    def __init__(self):
        print("Classifier Init Function")
        self.tree = None

        pass
    
    # it runs when the pacman has either won or lost. (when the game finishes)
    def reset(self):
        print("Classifier Reset Function")
        pass
    
    # runs once
    def fit(self, data, target):
        print("Classifier Fit Function")

        lst= [i for i in range(25)]
        # print(lst)
        # getProbabilityArray(data, target)
        self.tree = DTL(data, target, lst, [], [])
        print(self.tree.show())

        
        


        # calcEntropy(data, target)
        # calcGain(data,target)

        # print("this is the data: " + str(data))
        # print("this is the target: " + str(target))
        pass

    # runs every step of the game
    def predict(self, data, legal=None):
        print("Classifier Predict Function")

        # # use of variable tree
        # print(self.tree.left)
        # # print(self.tree.right)
        # print(self.tree.value)

        currentNode = self.tree

        # print(currentNode.show())


        while (currentNode.left is not None or currentNode.right is not None):

            if (data[currentNode.attribute] == 0):
                currentNode = currentNode.left
            elif (data[currentNode.attribute] == 1):
                currentNode = currentNode.right
        
        print("this is the final prediction: ", currentNode.value)
        return currentNode.value

        
