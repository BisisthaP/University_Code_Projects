#name - Bisistha Patra 
#Student ID - 24159091

#Task 2 
import random 
import csv 
import matplotlib.pyplot as plt

class Animal:
    def __init__(self, x, y):
        self.x = x #horizontal position
        self.y = y #vertical position 

#both Prey and Predators will inherit from the Animal base class 
class Prey(Animal):
    pass

#difference between preys and predators is the energy levels 
class Predator(Animal):
    #e = energy level
    def __init__(self, x, y, e):
        super().__init__(x, y) #call the base class constructor
        self.energy = e

#to keep the cost for moving diagonally the same as moving horizontally or vertically,
# we will use chessboard distance (Chebyshev distance)
def dist(x1,y1,x2,y2):
    return max(abs(x1 - x2), abs(y1 - y2))
    #max of the absolute differences in x and y coordinates 

#all of this is same as task 1 - no changes made (code lines above this comment)
