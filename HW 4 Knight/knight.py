from itertools import product, combinations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
from collections import Counter
# import os

# os.chdir(os.path.join(os.getcwd(), 'Desktop', 'ORIE', 'HW 4 Knight'))

def generateStates():
    states = {}
    positions = list(i for i in range(1,10) if i!= 5)
    index = 1
    for i in positions:
        for j in positions:
            for k in positions:
                for l in positions:
                    if i not in (j,k,l) and j not in (k,l) and k != l and i < j and k < l:
                        states[index] = (i,j,k,l)
                        index += 1
    return states


def isKnightMove(source, destination):
    mapping = {1: (6, 8), 2: (7, 9), 3: (4, 8), 4: (3, 9), 5: (), 6: (1, 7), 
                7: (2, 6), 8: (1, 3), 9: (2, 4)}
    return destination in mapping[source]

def possible(src, dst):
    """
    possible that src can transition to dst iff only one entry is different or two entries are different 
    the different entry is a knight move
    """
    diff = [i for i in range(4) if src[i] != dst[i]]
    if len(diff) == 1:
        # print(src[diff[0]], dst[diff[0]])
        return isKnightMove(source=src[diff[0]], destination=dst[diff[0]])
    elif len(diff) == 2 and (diff == [0,1] or diff == [2,3]):
        nodes, freqs = list(zip(*
            Counter( (src[diff[0]], src[diff[1]], dst[diff[0]], dst[diff[1]]) ).most_common()
        ))
        if freqs[0] == 2:
            return isKnightMove(source=nodes[1], destination=nodes[2])
        return False
    return False

# assert possible( (4,8,1,2), (4,8,2,6) ) == True
# assert possible( (4,8,1,2), (4,8,1,7) ) == True 
# assert possible( (4,8,1,2), (4,8,1,8) )  == False 
# assert possible( (4,8,1,2), (4,8,1,9) )  == True
# assert possible( (4,8,1,2), (4,8,2,9) )  == False
# assert possible( (1,3,7,9), (7,9,1,3) )  == False
# assert possible( (1,3,7,9), (1,8,7,9) )  == True
# assert possible( (1,3,7,9), (1,3,4,7) )  == True
# assert possible( (1,3,7,9), (3,6,7,9) )  == True
# assert possible( (1,3,7,9), (1,3,7,9) )  == False

# def debug():
#     S = generateStates()
#     x = [ [0 for _ in range(len(S))] for _ in range(len(S)) ]
#     for s in S:
#         for s_ in S:
#             if possible(S[s], S[s_]):
#                 x[s-1][s_-1] = 1
#     x = pd.DataFrame(x, index=S.keys(), columns=S.keys())
#     print(x.loc[1])
#     print(S[1], S[5])

def f(s):
    """returns the flow"""
    if s == (1,3,7,9): # source
        return 1
    elif s == (7,9,1,3): # sink
        return -1
    return 0

S = generateStates()
# Let x_ss' be 1 if there's a transition from state s to state s' and 0 otherwise

model = gp.Model("knight")

x = []
for s in S:
    for s_ in S:
        if possible(S[s], S[s_]):
            x.append( (s,s_) ) # model.addVar(vtype=GRB.BINARY, name="x_%s_%s" % (s, s_))
x = model.addVars(x, vtype=GRB.BINARY, name="x")

for s in S:
    model.addConstr( gp.quicksum(x[s, s_] for s_ in S if possible(S[s], S[s_])) - 
                     gp.quicksum(x[s_, s] for s_ in S if possible(S[s_], S[s])) == f(S[s]) 
                   )

model.setObjective( gp.quicksum(x[s, s_] for s in S for s_ in S if possible(S[s], S[s_])) , GRB.MINIMIZE)

model.optimize()

print('Optimal value:', model.objVal)

adjacencyList = {}
for v in model.getVars():
    if v.x > 0:
        indices = v.varName.replace('x', '').replace('[', '').replace(']', '').split(',')
        # print(v.varName, '=', v.x)
        # print(int(indices[0]), int(indices[1]))
        # print(S[int(indices[0])], S[int(indices[1])])
        # print(S[int(indices[0])], S[int(indices[1])], possible(S[int(indices[0])], S[int(indices[1])]))
        adjacencyList[S[int(indices[0])]] = S[int(indices[1])]

start = (1,3,7,9)
path = [start]
while path[-1] in adjacencyList: #should end at (7,9,1,3):
    path.append(adjacencyList[path[-1]])
    print(path[-1])


import matplotlib.pyplot as plt
dx, dy = (0.015, 0.015)

x = np.arange(0, 3, 0.015)
y = np.arange(0, 3, 0.015)

(X, Y) = np.meshgrid(x, y)
extent = (np.min(x), np.max(x), np.min(y), np.max(y))

z1 = np.add.outer(range(3), range(3)) % 2

white = plt.imread('dad.png')
black = plt.imread('father.jpeg')
def plotDad(x, y, w=True):
    plt.imshow(white if w else black, extent=[x-0.3, x+0.3, y-0.3, y+0.3], alpha=1, zorder=1)

plotDad(0.5, 0.5)
plotDad(2.5, 0.5)
plotDad(0.5, 2.5, False)
plotDad(2.5, 2.5, False)

plt.imshow(z1, cmap='binary_r', interpolation='nearest', extent=extent, alpha=1)
plt.show()
        




