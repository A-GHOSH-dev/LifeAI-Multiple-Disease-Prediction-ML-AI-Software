import random
import sys
import math
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
from sklearn.utils import shuffle


class whale:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def checkBounds(w):
    w.x = min(w.x,math.pow(2,15))
    w.x = max(w.x,math.pow(2,-5))
    w.y = min(w.y,2)
    w.y = max(w.y,math.pow(2,-15))
    return w
    
def generatePopulation(count):
    whales = []
    for i in range(0,count):
        x = random.uniform(math.pow(2,-5),1)
        y = random.uniform(math.pow(2,-3),1)
        whales.append(whale(x,y))
    return whales


def fitness(data_scaled, target, x,y):
    clf = svm.SVC(kernel='rbf', C=x, gamma=y)
    scores = cross_val_score(clf,data_scaled, target, cv=10)
    return scores.mean()

def findBestWhale(data_scaled, target, whales):
    bestFitness = -sys.maxsize
    bestWhale = -1
    for w in whales:
        currFitness = fitness(data_scaled, target, w.x,w.y)
        if(currFitness > bestFitness):
            bestFitness = currFitness
            bestWhale = w
    return bestWhale, bestFitness


# def readCancerDataset():
#     dataset = pd.read_csv('LC.csv', header=None)
#     dataset[10] = dataset[10].replace(2,0)
#     dataset[10] = dataset[10].replace(4,1)
#     dataset = dataset[dataset[6]!='?']
#     data = dataset.iloc[:, 1:-1].values
#     target = dataset.iloc[:, -1].values
#     return (data, target)

# def readDiabetesDataset():
#     dataset = pd.read_csv('diabetes.csv')
#     data = dataset.iloc[:,:-1].values
#     target = dataset.iloc[:,-1].values
#     return (data,target)

# def readParkinsonsDataset():
#     dataset = pd.read_csv('parkinsons.csv')
#     dataset = dataset.drop('name',1)
#     data = dataset.drop('status',1).values
#     target = dataset.iloc[:,16].values
#     return (data, target)

def readThyroidDataset():
    dataset = pd.read_csv('thyroid.csv', header=None)
    dataset = shuffle(dataset)
    target = dataset.iloc[:,-1]
    data = dataset.iloc[:, :-1]
    return (data, target)
    
    
def optimize(data_scaled, target, whales, maxIter):
    bestWhale, bestFitness = findBestWhale(data_scaled, target, whales)
    for i in range(0, maxIter):
        print(f'Running iteration {i+1}...')
        for w in whales:
            a = 2 - 2*i/maxIter
            r = random.uniform(0,1)
            A = 2*a*r - a
            C = 2*r
            b = 2
            l = 2*random.uniform(0,1) - 1
            p = random.uniform(0,1)
            
            if(p<0.5):
                if(abs(A)<1):
                    targetWhale = bestWhale
                else:
                    randPos = random.randint(0,len(whales)-1)
                    targetWhale = whales[randPos]
                    
                Dx = C*targetWhale.x - w.x
                Dy = C*targetWhale.y - w.y
                w.x = targetWhale.x - A*Dx
                w.y = targetWhale.y - A*Dy
                    
            else:
                Dx = bestWhale.x - w.x
                Dy = bestWhale.y - w.y
                w.x = Dx*math.exp(b*l)*math.cos(2*math.acos(-1.0)*l) + bestWhale.x
                w.y = Dy*math.exp(b*l)*math.cos(2*math.acos(-1.0)*l) + bestWhale.y
            
            w = checkBounds(w)
    
        bestWhale, bestFitness = findBestWhale(data_scaled, target, whales)
        print(round(bestWhale.x,3), round(bestWhale.y,3), round(bestFitness,5))
    return bestWhale, bestFitness

def main(popsize=10, maxIter=250):
    # data, target = readCancerDataset()
    data, target = readThyroidDataset()
    data_scaled = preprocessing.scale(data)
    whales = generatePopulation(popsize)
    params, accuracy = optimize(data_scaled, target, whales, maxIter)
    
main()
