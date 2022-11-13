from itertools import combinations, product, permutations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np


def debug(n, m, S, r, t, gurobiSolution):

    # let x_ij = 1 if slab j is used for product i
    # an optimal solution is: 

    # x_12 = 1, x_23 = 1, x_34 = 1

    x = [ [0 for j in range(m)] for i in range(n) ]
    x = pd.DataFrame( x, index = [f'Product {i}' for i in range(1, n+1)], columns = [f'Slab {j}' for j in range(1, m+1)] )

    for i, j in gurobiSolution:
        x.iloc[i-1, j-1] = 1

    # every product must be produced from only one slab
    # sum_j x_ij = 1 for all i
    for i in range(1, n+1):
        if sum(x.iloc[i-1][j-1] for j in range(1, m+1)) != 1:
            print(f'Product {i} is not produced from only one slab')


    # every slab can be used for at most one product
    # sum_i x_ij <= 1 for all j
    for j in range(1, m+1):
        if sum(x.iloc[i-1, j-1] for i in range(1, n+1)) > 1:
            print(f'Slab {j} is used for more than one product')
    
    # every product must be produced on a slab that is in the set of slabs that it can be produced on
    # x_ij = 0 for all i, j such that j not in S_i
    for i in range(1, n+1):
        for j in range(1, m+1):
            if j not in S[i] and x.iloc[i-1, j-1] != 0:
                print(f'Product {i} is produced on slab {j} which is not in its set of slabs')


    R = sum( x.iloc[i-1, j-1] * t[j] for i in range(1, n+1) for j in S[i] )

    
    # Let T represent the sum of deductions when slab j is used for product i due to the previous i-1 products that have been created
    # T = 0
    # for i in range(1, n+1):
    #     for j in S[i]:
    #         for k in range(1, n+1):
    #             for l in S[k]:
    #                 if (r[j] == r[l] and t[j] < t[l]):
    #                     T += x.iloc[i-1, j-1] * x.iloc[k-1, l-1]


    
    # Let y_ijkm = 1 if slab j is used for product i and slab m is used for product k
    y = [ [np.zeros((n,m)) for j in range(m) ] for i in range(n) ]
    y = pd.DataFrame( y, index = [f'Product {i}' for i in range(1, n+1)], columns = [f'Slab {j}' for j in range(1, m+1)] )

    for i in range(1, n+1):
        for j in S[i]:
            for k in range(i+1, n+1):
                for l in S[k]:
                    if i!=k and j!=m and (r[j] == r[l] and t[j] < t[l]):
                        y.iloc[i-1, j-1][k-1, l-1] = x.iloc[i-1, j-1] * x.iloc[k-1, l-1]

    # # i,j = 1,2
    # # i,j = 2,3
    # # i,j = 3,4
    # # print(y.iloc[i-1, j-1])


    # y_ijkm <= x_ij for all i, j, k, m --> if slab j is not used for product i, then for all (k, m), slab m cannot be used for product k
    # y_ijkm <= x_km for all i, j, k, m --> if slab k is not used for product m, then for all (i, j), slab j cannot be used for product i
    # y_ijkm >= x_ij + x_km - 1 for all i, j, k, m --> the only way for slab j to be used for product i and slab m to be used for product k
    # is if slab j is used for product i and slab m is used for product k
    for i in range(1, n+1):
        for j in S[i]:
            for k in range(i+1, n+1):
                for l in S[k]:
                    if i!=k and j!=l and (r[j] == r[l] and t[j] < t[l]):
                        if y.iloc[i-1, j-1][k-1, l-1] > x.iloc[i-1, j-1]:
                            print(f'Product {i} is not used for slab {j}, but slab {l} is used for product {k}')
                        if y.iloc[i-1, j-1][k-1, l-1] > x.iloc[k-1, l-1]:
                            print(f'Product {k} is not used for slab {l}, but slab {j} is used for product {i}')
                        if y.iloc[i-1, j-1][k-1, l-1] < x.iloc[i-1, j-1] + x.iloc[k-1, l-1] - 1:
                            print('i =', i, 'j =', j, 'k =', k, 'l =', l)
                            print('x_ij =', x.iloc[i-1, j-1], 'x_kl =', x.iloc[k-1, l-1], 'y_ijkl =', y.iloc[i-1, j-1][k-1, l-1])
                            print()

    T = 0
    for i in range(1, n+1):
        for j in S[i]:
            for k in range(i+1, n+1):
                for l in S[k]:
                    if i!=k and j!=m and (r[j] == r[l] and t[j] < t[l]):
                        T += y.iloc[i-1,j-1][k-1,l-1]
    print(x)
    print('R=', R, ' ', 'T=', T, ' ', '(R - T)=', R - T)
    if (R - T == 4 and R == 5 and T == 1) or (R - T == 3 and R == 6 and T == 3):
        print('YAY')


def run(n, m, S, r, t):
    model = gp.Model()
    model.Params.LogToConsole = 1
    model.Params.OutputFlag = 1
    x = model.addVars( list(product(range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'x' )
    # y = model.addVars( list(product(range(1, n+1), range(1, m+1), range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'y' )


    # 1. every product must be produced from only one slab
    # sum_j x_ij = 1 for all i
    for i in range(1, n+1):
        model.addConstr( gp.quicksum( x[i,j] for j in range(1, m+1) ) == 1 )

    # 2. every slab can be used for at most one product
    # sum_i x_ij <= 1 for all j
    for j in range(1, m+1):
        model.addConstr( gp.quicksum( x[i,j] for i in range(1, n+1) ) <= 1 )

    # 3. every product must be produced on a slab that is in the set of slabs that it can be produced on
    # x_ij = 0 for all i, j such that j not in S_i
    for i in range(1, n+1):
        for j in range(1, m+1):
            if j not in S[i]:
                model.addConstr( x[i,j] == 0 )

    # Let R represent the cost of producing every product, not considering the deduction of previous i-1 products that have been created
    R = gp.quicksum( x[i, j] * t[j] for i in range(1, n+1) for j in range(1, m+1) )


    # T = 0
    # for i in range(1, n+1):
    #     for j in S[i]:
    #         for k in range(i+1, n+1):
    #             for u in S[k]:
    #                 if (r[j] == r[u] and t[j] < t[u]) and (i != k) and (j != u):
    #                     T += x[i,j] * x[k,u]

    # y_ijkm <= x_ij for all i, j, k, m --> if slab j is not used for product i, then for all (k, m), slab m cannot be used for product k
    # y_ijkm <= x_km for all i, j, k, m --> if slab k is not used for product m, then for all (i, j), slab j cannot be used for product i
    # y_ijkm >= x_ij + x_km - 1 for all i, j, k, m --> the only way for slab j to be used for product i and slab m to be used for product k

    y = model.addVars( list(product(range(1, n+1), range(1, m+1), range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'y' )
    for i in range(1, n+1):
        for j in S[i]:
            for k in range(i+1, n+1):
                for m in S[k]:
                    if i!= k and j != m:
                        model.addConstr( y[i,j,k,m] <= x[i,j] )
                        model.addConstr( y[i,j,k,m] <= x[k,m] )
                        model.addConstr( y[i,j,k,m] >= x[i,j] + x[k,m] - 1 )
    T = 0
    for i in range(1, n+1):
        for j in S[i]:
            for k in range(i+1, n+1):
                for u in S[k]:
                    if (r[j] == r[u] and t[j] < t[u]) and i!= k and j != m:
                        T += y[i,j,k,u]


    model.setObjective( R - T, GRB.MINIMIZE )

    model.optimize()

    print('Optimal value:', model.objVal)

    for v in model.getVars():
        if v.x > 0:
            print(v.varName, '=', v.x)
    

n = 3 # number of products i ranges from 1 to n
m = 8 # number of slabs ranges from 1 to m
# Si is the set of slabs that product i can be produced on
S = {
    1: [2, 5],
    2: [3, 7],
    3: [4]
}
# stack number that slab j is in
r = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2
}

# number of slabs on top of slab j
t = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 3
}

# n = 2
# m = 4
# S = { 1: [1,3], 2: [2]}
# r = { 1: 1, 2: 1, 3: 2, 4: 2}
# t = { 1: 0, 2: 1, 3: 0, 4: 1}

# n = 4
# m = 9
# S = {
#     1: [7, 8],
#     2: [3, 4],
#     3: [7],
#     4: [9]
# }
# r = {i: 1 for i in range(1, m+1)}
# t = {i: i-1 for i in range(1, m+1)}


# df = pd.read_csv('Dataset_slab_stack.csv')
# n = 75
# m = 250
# S = {row['Product_number']: row.iloc[1:].to_numpy() for index, row in df.iterrows()}
# r = { range(1,61) : 1, range(61,161) : 2, range(161, 201) : 3, range(201, 251) : 4 }
# r = {s: stackNum for slab, stackNum in r.items() for s in slab}
# t = { range(1,61) , range(61,161) , range(161, 201) , range(201, 251)  }
# t = {i: i-min(slab) for i in range(1,251) for slab in t if i in slab}

# run(n, m, S, r, t)


# debug(n, m, S, r, t, gurobiSolution= [ (1, 2), (2, 3), (3, 4) ])
# debug(n, m, S, r, t, gurobiSolution= [(1,5),(2,7),(3,4)] )
# debug(n, m, S, r, t, gurobiSolution= [ (1,1), (2,2) ])



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def label(ax, rect, text):
    rx, ry = rect.get_xy()
    cx = rx + rect.get_width()/2.0
    cy = ry + rect.get_height()/2.0
    ax.annotate(text, (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')

def stack(ax, x_start, bottom_to_top_list):
    rectangles = [ Rectangle((x_start, i),1,1, edgecolor='r', facecolor='b') \
                    for i, slab in enumerate(bottom_to_top_list) ]
    for rect, slab in zip(rectangles, bottom_to_top_list):
        ax.add_patch(rect)
        label(ax, rect, slab)

# fig, ax = plt.subplots()
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)

# stack(ax, 0, [4, 3, 2, 1])
# stack(ax, 2, [8, 7, 6, 5])
# stack(ax, 0, list(range(9, 0, -1)))

# plt.show()