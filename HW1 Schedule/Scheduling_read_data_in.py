# -*- coding: utf-8 -*-

# with open('gurobi.lic', 'r') as f:
#     lic = f.readlines()

# WLSACCESSID = lic[-3].replace('\n', '').replace('WLSACCESSID=', '')
# WLSSECRET = lic[-2].replace('\n', '').replace('WLSSECRET=', '')
# LICENSEID = int( lic[-1].replace('\n', '').replace('LICENSEID=', '') )

from itertools import combinations, product, permutations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

import os
os.chdir(os.getcwd() + '/Desktop')

df=pd.read_csv('sched_med_proc_times.csv', header=None)
df2=pd.read_csv('schedmed_prec.csv') # precedence constraints
n=10#df.shape[0]          # Number of Jobs
m=2            # Number of machines
# Need to find a way to read the data in and convert it into the appropriate form
prec_dict={}  # Should have keys 1....N and each key is mapped to the set of preceding jobs. 
# The "set" needs to be a list so Gurobi can iterate over it.
proc_dict={} # Same keys as prec_dict, but mapped to processing times instead.
T= df.iloc[:n, 1].sum()
for j in range(n):
    proc_dict[j]=df.iloc[j,1]
    prec_dict[j]=list(df2.iloc[j].dropna().to_numpy() -1 )[1:]



# given m machines, n jobs, a set of precedence constrains (prec_dict), and processing times (proc_dict)
# find the minimum makespan ( i.e. minimize the largest completion time of any job) where 
# each machine can only work on one job at a time, and each job must be completed before the next job can start.
# The precedence constraints are given in the form of a dictionary where the keys are the jobs and the values are the
# set of jobs that must be completed before the key job can start.
# The processing times are given in the form of a dictionary where the keys are the jobs and the values are the
# processing times of the jobs.

def Scheduling(m,n,prec_dict,proc_dict,T):
    """
    Parameters
    ----------
    m : int - number of machines
    n : int - number of jobs
    prec_dict : dictionary - keys are jobs, values are the set of jobs that must be completed before the key job can start
    Example: prec_dict[3] = [1,2] means that jobs 1 and 2 must be completed before job 3 can start
    proc_dict : dictionary - keys are jobs, values are the processing times of the jobs
    Example: proc_dict[3] = 5 means that job 3 takes 5 time units to complete
    T : int - upper bound on the makespan
    """
    # p[j] is the processing time of job j
    model = gp.Model()
    list_of_ait = list(product(range(n), range(m), range(T)))

    x_ait = model.addVars(list_of_ait, vtype=GRB.BINARY, name="x_ait")

    #1. each job can only be assigned to one machine at one time
    model.addConstrs( 
        gp.quicksum(x_ait[a,i,t] for i in range(m) for t in range(T)) == 1 for a in range(n) 
        )

    #2. for every machine at time t, it can be assigned at most one job
    model.addConstrs(
        gp.quicksum(x_ait[a,i,t] for a in range(n)) <= 1 for i in range(m) for t in range(T)
        )

    #3. for each job a' must be completed before job a can start
    # a ≺ a_ to mean that job a must be completed before task a_ can start
    # preceding_pairs = [ i for j in [ list(product( value, [key]) ) for key, value in prec_dict.items() ] for i in j]
    # for a, a_ in preceding_pairs:
    #     for t in range(T):
    #         prev = dp.iloc[a-1].to_numpy()
    #         later = dp.iloc[a_-1].to_numpy()
    #         if sum( prev[i][t_] for i in range(m) for t_ in range(t)) < sum( later[i][t_] for i in range(m) for t_ in range(min(T, t + proc_dict[a])) ):
    #             print(f'job {a_} cannot begin while job {a} + p({a}) = {proc_dict[a] + t}' )

    preceding_pairs = [ i for j in [ list(product( value, [key]) ) for key, value in prec_dict.items() ] for i in j]
    for a, a_ in preceding_pairs:
        for t in range(T):
            model.addConstr(
                gp.quicksum(x_ait[a,i,t_] for i in range(m) for t_ in range(t)) >= gp.quicksum(x_ait[a_,i,t_] for i in range(m) for t_ in range(min(T, t + proc_dict[a])))
                )

        

    # z = model.addVars(list_of_ait, name='auxillary')
    #4. minimize the makespan
    #makespan = max ( k + proc_dict[i+1] for i in range(n) for j in range(m) for k in range(T) if dp.iloc[i,j][k] == 1 )
    # for a in range(n):
    #     for i in range(m):
    #         for t in range(T):
    #             model.addConstr( z[a,i,t] == x_ait[a,i,t] * (t+ proc_dict[a]) )

    # y = model.addVar( vtype=GRB.INTEGER, name="makespan" )
    # model.addConstr( y == gp.max_(z) )

    #4. job a cannot begin on a machine i while job a' + p(a') is still being processed on machine i
    #from itertools import permutations
    # jobs = list( permutations( prec_dict.keys(), 2) )
    # for i in range(m):
    #     for a, a_ in jobs:
    #         for t in range(T):
    #             if dp.iloc[a_-1,i][t] == 1:
    #                 arr = dp.iloc[a-1,i]
    #                 arr_ = dp.iloc[a_-1,i]
    #                 if sum( [arr[t_] + arr_[t_] for t_ in range(t, +proc_dict[a_])] ) > 1:
    #                     print(f'job {a} cannot begin on a machine {i+1} while job {a_} + p({a_}) = {proc_dict[a_] + t}')

    jobs = list(permutations(range(n), 2))
    model.addConstrs(
        ( (x_ait[a_,i,t] == 1) >> ( gp.quicksum( x_ait[a,i,t_] + x_ait[a_,i,t_] for t_ in range(t, min(T, t+proc_dict[a_])) ) <= 1 ) ) for a, a_ in jobs for i in range(m) for t in range(T)
    )
    """
    translate this to latex

    $$ if \ x_{a_,i,t} = 1, then \sum_{t_ = t}^{t + p(a_)} x_{a,i,t_} + x_{a_,i,t_} \leq 1 $$

    """


    model.setObjective(gp.quicksum( x_ait[a,i,t] * (t+ proc_dict[a]) for a in range(n) for i in range(m) for t in range(T) ), GRB.MINIMIZE) 

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        print('Optimal objective: %g' % model.objVal)
        print('Optimal solution:')
        makespan = float('-inf')
        for v in model.getVars():
            if v.x > 0:
                print('%s %g' % (v.varName, v.x))
                makespan = max(makespan, int(v.varName.split('[')[1].split(']')[0].split(',')[2]) + proc_dict[int(v.varName.split('[')[1].split(']')[0].split(',')[0])])
        print(f'makespan: {makespan}')
    else:
        print('No solution')





# m = 3
# n = 7
# prec_dict = {1:[], 2:[], 3:[], 4: [1,3], 5:[1,2], 6:[4], 7:[]}
# proc_dict = {1: 3, 2: 1, 3:1, 4:1, 5:5, 6:5, 7:5}
# T = sum(proc_dict.values())
# prec_dict = {key-1: [i-1 for i in value] for key, value in prec_dict.items()}
# proc_dict = {key-1: value for key, value in proc_dict.items()}

Scheduling(m,n,prec_dict,proc_dict,T)

"""
#n by m by T
dp = [ [np.zeros(T).astype(int) for i in range(m)] for j in range(n) ]
dp = pd.DataFrame( dp, columns = [f'Machine {i}' for i in range(1, m+1)], index = [f'Job {i}' for i in range(1, n+1)] )

# x(a,i, t) = 1 if task a is assigned to machine i at time t
# 0<=a<=n-1, 0<=i<=m-1, 0<=t<=T

# a valid solution is:
# x(2, 1, 0) , x(3, 1, 1), x(4, 1, 3), x(6, 1, 4)
# x(1, 2, 0) , x(5, 2, 3),
# x(7, 3, 0)
# for i,j,k in [ (2, 1, 0), (3, 1, 1), (4, 1, 3), (6, 1, 4), (1, 2, 0), (5, 2, 3), (7, 3, 0) ]:
#     dp.iloc[i-1,j-1][k] = 1

# for i,j,k in [ [1,3,0], [2,2,0], [3,1,0], [4,1,1], [5,2,1], [6,1,2], [7,3,1] ]:
#     dp.iloc[i-1,j-1][k] = 1

# for i,j,k in [ [1,2,0], [2,1,0], [3,3,0], [4,1,1], [5,3,1], [6,1,2], [7,2,3] ]:
#     dp.iloc[i-1,j-1][k] = 1



arr = [ [0,1,0], [1,0,0], [2,2,0], [3,0,1], [4,1,3], [5,0,2], [6,2,1] ]
arr = [[1, 2, 0], [2, 1, 0], [3, 3, 0], [4, 1, 1], [5, 2, 3], [6, 1, 2], [7, 3, 1]]
# ADD 1 to first two elements of each subarray
# arr = [ [i+1, j+1, k] for i,j,k in arr ]
for i,j,k in [[1, 2, 0], [2, 1, 0], [3, 3, 0], [4, 1, 1], [5, 2, 3], [6, 1, 2], [7, 3, 1]]:
    dp.iloc[i-1,j-1][k] = 1



# Let's build the constraints:
# row sum of dp 
dp.sum(axis=1).apply( lambda x: sum(x) )

#1. each job can only be assigned to one machine at one time
for i in range(n):
    arr = dp.iloc[i].to_numpy() # m by T
    if sum( arr[i][j] for i in range(m) for j in range(T) ) != 1:
        print('each job can only be assigned to one machine at one time')

# column sum of dp
dp.sum(axis=0).to_numpy()

#2. for every machine at time t, it can be assigned at most one job
for j in range(m): 
    for t in range(T):
        arr = [x[t] for x in dp.iloc[:,j].to_numpy()]
        if sum(arr) > 1:
            print('for every machine at time t, it can be assigned at most one job')

#3. for each job a' must be completed before job a can start
# a ≺ a_ to mean that job a must be completed before task a_ can start
# Example: if a_ = 4, a = 1 and p(1) = 3, then 
# then sum over i, t' of x(4, i, t') >= sum over i, t' of x(1, i, t) + p(1)

preceding_pairs = [ i for j in [ list(product( value, [key]) ) for key, value in prec_dict.items() ] for i in j]

for a, a_ in preceding_pairs:
    for t in range(T):
        prev = dp.iloc[a-1].to_numpy()
        later = dp.iloc[a_-1].to_numpy()
        if sum( prev[i][t_] for i in range(m) for t_ in range(t)) < sum( later[i][t_] for i in range(m) for t_ in range(min(T, t + proc_dict[a])) ):
            print(f'job {a_} cannot begin while job {a} + p({a}) = {proc_dict[a] + t}' )
        # if (a, t) == (1, 0) and a_ ==4:
        #     print(prev)
        #     print(later)
            # print( sum( prev[i][t_] for i in range(m) for t_ in range(min(T, t + proc_dict[a]))) )
            # print( sum( later[i][t_] for i in range(m) for t_ in range(t) ) )

#4 job a cannot begin on a machine i while job a' + p(a') is still being processed on machine i
# Example: if a = 7 and a' = 1 with p(1) = 3, then 
# x(1, i, t) + x(7, i, t) <= 1 for every i, t'<=t+3 (t' = t, t+1, t+2, t+3)
from itertools import permutations
jobs = list( permutations( prec_dict.keys(), 2) )
for i in range(m):
    for a, a_ in jobs:
        for t in range(T):
            if dp.iloc[a_-1,i][t] == 1:
                arr = dp.iloc[a-1,i]
                arr_ = dp.iloc[a_-1,i]
                # if (a, a_) == (7,1) and i == 2:
                #     print( t, [arr[t_] + arr_[t_] for t_ in range(t, t+proc_dict[a_])])
                if sum( [arr[t_] + arr_[t_] for t_ in range(t, min(T, t+proc_dict[a_]))] ) > 1:
                    print(f'job {a} cannot begin on a machine {i+1} while job {a_} + p({a_}) = {proc_dict[a_] + t}')




makespan = 0
for i in range(n):
    for j in range(m):
        for k in range(T):
            if dp.iloc[i,j][k] == 1:
                makespan += k + proc_dict[i + 1]
    


print(makespan)
"""


"""
Translate to latex:

The following code cells illustrate how I came up with valid constraints based on the decision variable:


$$ X_{ait} = \begin{cases} 1 & \text{if job a is assigned to machine i at time t} \\ 0 & \text{otherwise} \end{cases}
\\ a \in \{1, 2, \dots, n\}, i \in \{1, 2, \dots, m\}, t \in \{0, 1, \dots, T\} $$


Note that as discussed in class, this formulation may be computationally expensive, as it will require 

$$ m \cdot n \cdot T $$ variables.  To come up with the constraints, I used the example in class where 
$$ m = 3, n = 7, T = 21 $$

Our first constraint is that each job can only be assigned to one machine at one time.  This is equivalent to the following constraint:

$$ \sum_{i=1}^m \sum_{t=0}^T X_{ait} = 1 \quad \forall a \in \{1, 2, \dots, n\} $$

The second constraint is that for every machine at time t, it can be assigned at most one job.  This is equivalent to the following constraint:

$$ \sum_{a=1}^n X_{ait} \leq 1 \quad \forall i \in \{1, 2, \dots, m\}, \forall t \in \{0, 1, \dots, T\} $$

The third constraint is that for each job a' must be completed before job a can start.  Denote a ≺ a' to mean that job a must be completed before task a' can start.  Denote p(a) to be the processing time of job a.  This is equivalent to the following constraint:

$$ \sum_{i=1}^m \sum_{t'=0}^{t-1} X_{a'it'} \leq \sum_{i=1}^m \sum_{t'=0}^{t+p(a)-1} X_{ait'} \quad \forall a ≺ a' \in \{1, 2, \dots, n\}, \forall t \in \{0, 1, \dots, T\} $$

The fourth constraint is that job a cannot begin on a machine i while job a' + p(a') is still being processed on machine i.  This is equivalent to the following:

$$ \sum_{t'=t}^{t+p(a')-1} X_{a'it'} + X_{ait'} \leq 1 \quad \forall perm(a, a') \in \{1, 2, \dots, n\}^2, \forall i \in \{1, 2, \dots, m\}, \forall t \in \{0, 1, \dots, T\} $$

Our objective function is to minimize the makespan, which is the time at which the last job is completed.  This is equivalent to the following:

$$ 


"""