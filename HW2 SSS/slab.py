from itertools import combinations, product, permutations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

def subsets(nums):
    res = [[]]
    for num in nums:
        res += [item + [num] for item in res]
    return res

n = 3 # number of products i ranges from 1 to n
m = 8 # number of slabs ranges from 1 to m
# Ci is the set of slabs that product i can be produced on
C = {
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

n = 2
m = 4
C = { 1: [1,3], 2: [2]}
r = { 1: 1, 2: 1, 3: 2, 4: 2}
t = { 1: 0, 2: 1, 3: 0, 4: 1}




def debug():
    n = 3 # number of products i ranges from 1 to n
    m = 8 # number of slabs ranges from 1 to m
    C = {
        1: [2, 5],
        2: [3, 7],
        3: [4]
    }
    r = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}
    t = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 3}

    # let x_ij = 1 if slab j is used for product i
    # an optimal solution is: 

    # x_12 = 1, x_23 = 1, x_34 = 1

    x = [ [0 for j in range(m)] for i in range(n) ]
    x = pd.DataFrame( x, index = [f'Product {i}' for i in range(1, n+1)], columns = [f'Slab {j}' for j in range(1, m+1)] )

    for i, j in [(1,5),(2,7),(3,4)]:
        x.iloc[i-1, j-1] = 1
    # for i, j in [ (1, 2), (2, 3), (3, 4) ]: 
    #     x.iloc[i-1, j-1] = 1

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
    # x_ij = 0 for all i, j such that j not in C_i
    for i in range(1, n+1):
        for j in range(1, m+1):
            if j not in C[i] and x.iloc[i-1, j-1] != 0:
                print(f'Product {i} is produced on slab {j} which is not in its set of slabs')



    R = sum( x.iloc[i-1, j-1] * t[j] for i in range(1, n+1) for j in C[i] )

    
    # Let T represent the sum of deductions when slab j is used for product i due to the previous i-1 products that have been created
    # T = 0
    # for i in range(1, n+1):
    #     for j in C[i]:
    #         for k in range(1, n+1):
    #             for u in C[k]:
    #                 if (r[j] == r[u] and t[j] < t[u]):
    #                     T += x.iloc[i-1, j-1] * x.iloc[k-1, u-1]


    
    # Let w_ijkm = 1 if slab j is used for product i and slab m is used for product k
    w = [ [np.zeros((n,m)) for j in range(m) ] for i in range(n) ]
    w = pd.DataFrame( w, index = [f'Product {i}' for i in range(1, n+1)], columns = [f'Slab {j}' for j in range(1, m+1)] )

    for i in range(1, n+1):
        for j in C[i]:
            for k in range(1, n+1):
                for m in C[k]:
                    # if i!=k and j!=m:
                        w.iloc[i-1, j-1][k-1, m-1] = x.iloc[i-1, j-1] * x.iloc[k-1, m-1]

    # # i,j = 1,2
    # # i,j = 2,3
    # # i,j = 3,4
    # # print(w.iloc[i-1, j-1])


    # w_ijkm <= x_ij for all i, j, k, m --> if slab j is not used for product i, then for all (k, m), slab m cannot be used for product k
    # w_ijkm <= x_km for all i, j, k, m --> if slab k is not used for product m, then for all (i, j), slab j cannot be used for product i
    # w_ijkm >= x_ij + x_km - 1 for all i, j, k, m --> the only way for slab j to be used for product i and slab m to be used for product k
    # is if slab j is used for product i and slab m is used for product k
    for i in range(1, n+1):
        for j in C[i]:
            for k in range(1, n+1):
                for m in C[k]:
                    if w.iloc[i-1, j-1][k-1, m-1] > x.iloc[i-1, j-1]:
                        print(f'Product {i} is not used for slab {j}, but slab {m} is used for product {k}')
                    if w.iloc[i-1, j-1][k-1, m-1] > x.iloc[k-1, m-1]:
                        print(f'Product {k} is not used for slab {m}, but slab {j} is used for product {i}')
                    if w.iloc[i-1, j-1][k-1, m-1] < x.iloc[i-1, j-1] + x.iloc[k-1, m-1] - 1:
                        print('i =', i, 'j =', j, 'k =', k, 'm =', m)
                        print('x_ij =', x.iloc[i-1, j-1], 'x_km =', x.iloc[k-1, m-1], 'w_ijkm =', w.iloc[i-1, j-1][k-1, m-1])
                        print()

    T = 0
    for i in range(1, n+1):
        for j in C[i]:
            for k in range(1, n+1):
                for u in C[k]:
                    if (r[j] == r[u] and t[j] < t[u]):
                        T += w.iloc[i-1,j-1][k-1,u-1]
    print(x)
    print('R=', R, ' ', 'T=', T, ' ', '(R - T)=', R - T)
    if (R - T == 4 and R == 5 and T == 1) or (R - T == 3 and R == 6 and T == 3):
        print('YAY')


def run(yes):
    global m
    model = gp.Model()
    x = model.addVars( list(product(range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'x' )
    # w = model.addVars( list(product(range(1, n+1), range(1, m+1), range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'w' )


    # 1. every product must be produced from only one slab
    # sum_j x_ij = 1 for all i
    for i in range(1, n+1):
        model.addConstr( gp.quicksum( x[i,j] for j in range(1, m+1) ) == 1 )

    # 2. every slab can be used for at most one product
    # sum_i x_ij <= 1 for all j
    for j in range(1, m+1):
        model.addConstr( gp.quicksum( x[i,j] for i in range(1, n+1) ) <= 1 )

    # 3. every product must be produced on a slab that is in the set of slabs that it can be produced on
    # x_ij = 0 for all i, j such that j not in C_i
    for i in range(1, n+1):
        for j in range(1, m+1):
            if j not in C[i]:
                model.addConstr( x[i,j] == 0 )

    # Let R represent the cost of producing every product, not considering the deduction of previous i-1 products that have been created
    R = gp.quicksum( x[i, j] * t[j] for i in range(1, n+1) for j in range(1, m+1) )


    # T = 0
    # for i in range(1, n+1):
    #     for j in C[i]:
    #         for k in range(1, n+1):
    #             for u in C[k]:
    #                 if (r[j] == r[u] and t[j] < t[u]):
    #                     T += x[i,j] * x[k,u]
    # w_ijkm <= x_ij for all i, j, k, m --> if slab j is not used for product i, then for all (k, m), slab m cannot be used for product k
    # w_ijkm <= x_km for all i, j, k, m --> if slab k is not used for product m, then for all (i, j), slab j cannot be used for product i
    # w_ijkm >= x_ij + x_km - 1 for all i, j, k, m --> the only way for slab j to be used for product i and slab m to be used for product k
    w = model.addVars( list(product(range(1, n+1), range(1, m+1), range(1, n+1), range(1, m+1))), vtype = GRB.BINARY, name = 'w' )
    for i in range(1, n+1):
        for j in C[i]:
            for k in range(1, n+1):
                for m in C[k]:
                    if i!= k and j != m:
                        model.addConstr( w[i,j,k,m] <= x[i,j] )
                        model.addConstr( w[i,j,k,m] <= x[k,m] )
                        model.addConstr( w[i,j,k,m] >= x[i,j] + x[k,m] - 1 )
    T = 0
    for i in range(1, n+1):
        for j in C[i]:
            for k in range(1, n+1):
                for u in C[k]:
                    if (r[j] == r[u] and t[j] < t[u]):
                        T += w[i,j,k,u]


    if yes:
        model.setObjective( R - T, GRB.MINIMIZE )

        model.optimize()

        print('Optimal value:', model.objVal)

        for v in model.getVars():
            if v.x > 0:
                print(v.varName, '=', v.x)
    

        

run(yes=True)

# for L in subsets([1, 2, 3, 4, 5, 6, 7, 8])[1:]: run(L, yes=True)

# debug()
# for L in subsets([1, 2, 3, 4, 5, 6, 7, 8])[1:]: debug(L)