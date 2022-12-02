import numpy as np
"""
https://cbom.atozmath.com/example/CBOM/Simplex.aspx?he=e&q=pd
Input:
m machines
for each machine i:
    di is the number of work cycles to be executed on machine i
    ti is the time it takes to execute a work cycle on machine i (hours)
    ai is the noise dosage that one work cycle on machine i inflict on worker
    ui is the maximum number of work cycles any worker can execute on machine i

H is the maximum number of hours a worker can work
z is the upper bound on how much noise dosage any schedule can have 

Output: find an assignment of work cycles to workers such that the maximum noise 
dosage inflicted on any worker is minimized
"""


n = 36
m = 25
H = 80
z = 145

arr = np.array([
1,	2,	41,	33,	43,	
2,	2,	33,	93,	29,	
3,	4,	9,  57,	10,	
4,	6,	9,	59,	17,	
5,	2,	6,	36,	23,	
6,	6,	41,	86,	49,	
7,	4,	12,	14,	30,	
8,	3,	17,	97,	53,	
9,	3,	29,	17,	39,	
10,	4,	17,	40,	21,	
11,	2,	17,	71,	39,	
12,	4,	26,	30,	57,	
13,	3,	37,	46,	21,	
14,	3,	12,	63,	37,	
15,	2,	16,	22,	11,	
16,	6,	57,	72,	33,	
17,	5,	10,	24,	37,	
18,	3,	39,	91,	27,	
19,	3,	42,	87,	54,	
20,	3,	35,	95,	46,	
21,	6,	35,	51,	45,	
22,	2,	24,	18,	54,	
23,	4,	20,	94,	27,	
24,	3,	24,	47,	29,	
25,	5,	58,	21,	50,	
]).reshape(25,5).T

d, t, alpha, u = arr[1], arr[2], arr[3], arr[4]

"""
Let s be the schedules for workers of how many work cycles they execute on each machine


xs = number of workers that work according to schedule s
s specifies the number of work cycles for each machine

Example:
s1 = {
    1: 2 // 2 cycles on machine 1
    2: 3 // 3 cycles on machine 2
}
xs1 = 4 // 4 workers on schedule s1



Objective:
min sum over s of (xs)


Constraints: 
Let wsi be the number of work cycles on machine i in schedule s <-- this is a parameter (input)
sum over s of (wsi * xs) >= di for all machiens i

xs>=0, xs is integer

translate to latex:

Let $s$ be the schedules for workers of how many work cycles they execute on each machine

$x_s$ is the number of workers that work according to schedule $s$

Objective:
minimize $\sum\limits_{s} x_s$

Constraints:
Let $w_{si}$ be the number of work cycles on machine $i$ in schedule $s$ <-- this is a parameter (input)
$\sum\limits_{s} w_{si} x_s \geq d_i$ \forall machines $i$


"""
import gurobipy as gp
from gurobipy import GRB
def Primal(m, d, S, w):
    """
    m: number of machines
    d: number of work cycles to be executed on each machine
    S: number of schedules
    w: number of work cycles on each machine in each schedule
    """
    model = gp.Model("Primal")
    x = model.addVars(list(range(1, S+1)), vtype=GRB.INTEGER, name="xs")
    model.setObjective(x.sum(), GRB.MINIMIZE)
    for i in range(m):
        model.addConstr(gp.quicksum(w[s][i] * x[s] for s in S) >= d[i])
    model.optimize()
    return model.objVal

S = 25
w = np.ones((S, m))
print(Primal(m, d, S, w))


import pandas as pd
pd.DataFrame(w, columns=[f'machine {i}' for i in range(1, m+1)], index=[f'schedule {i}' for i in range(1, S+1)])



"""
We now want to deal with the fact that we have a large number of variables by only generating a few
We need to convert this to a maximization problem:


Objective:
max -(sum over s of (xs) )

Constraints:
- sum over s of (wsi * xs) <= -di for all machines i,

xs>=0, xs is integer


Dual of LP-relaxation:

Objective:
min - sum over i of (di * yi)

Constraints:
sum over i of ( -wsi * yi) >= -1 for all schedules s

yi >= 0 for all machines i


Rewrite the dual as a maximization problem:

Objective:
max sum over i of (di * yi)

Constraints:
sum over i of (wsi * yi) <= 1 for all schedules s

yi >= 0 for all machines i

convert to latex:

Objective:
maximize $\sum\limits_{i} d_i y_i$

Constraints:
$\sum\limits_{i} w_{si} y_i \leq 1$ $\forall$ schedules $s$

$y_i \geq 0$ $\forall$ machines $i$
"""

def Dual(m, d, S, w):
    """
    m: number of machines
    d: number of work cycles to be executed on each machine
    S: number of schedules
    w: number of work cycles on each machine in each schedule
    """
    model = gp.Model("Dual")
    y = model.addVars(list(range(1, m+1)), vtype=GRB.CONTINUOUS, name="yi")
    model.setObjective(gp.quicksum(d[i-1] * y[i] for i in range(1, m+1)), GRB.MAXIMIZE)
    for s in range(S):
        model.addConstr(gp.quicksum(w[s][i-1] * y[i] for i in range(1, m+1)) <= 1)
    model.optimize()
    yi_star = []
    for v in model.getVars():
        if v.x > 0:
            print(v.varName, '=', v.x)
            yi_star.append(v.x)
    return model.objVal, yi_star

print(Dual(m, d, S, w))

"""
Idea now:
- Solving the primal with only a subset of the variables is the same as solving the dual with only a subset of the constraints
- We get an optimal dual solution for the relaxation that only has a subset of the constraints
- call this optimal dual solution yi*
- Now we want to check whethe the constraints we satisfied for all schedules, and if not produce the scheduel for which it is violated

We will use the dual with a subset of schedules

for which schedule we don't have in our solution, is our constraint > 1 ? if not

is there a schedule that does not hold for yi* ? if not, we have a valid solution

If we do find a schedule which violates the constraint, we add it to our solution and repeat the process


maximize the sum over i of (wsi * yi*) <-- this is now input to this problem
s.t. wsi corresponds to a valid schedule <-- these are decision variables


z is the upper bound on how much noise dosage any schedule can have 
Decision variables:
wi = number of work cycles on machine i

Objective:
max sum over i of (wi * yi*)

Constraints:
sum over i of ai * wi <= z <-- noise dosage for this schedule

wi <= ui for all machines i

sum over i of ti * wi <= H <-- time in hours for this schedule

wi >= 0 for all machines i

If objective value of an optimal solution <=1, then we do not have any violated constraints in the dual
-- means yi* is an optimal solution even with all constraints.

Otherwise, we found a schedule that corresponds to a violated constraint in dual

Translate to latex:
Idea now:
- Solving the primal with only a subset of the variables is the same as solving the dual with only a subset of the constraints
- We get an optimal dual solution for the relaxation that only has a subset of the constraints
- call this optimal dual solution $y_i^*$
- Now we want to check whether the constraints we satisfied for all schedules, and if not produce the scheduel for which it is violated

Decision variables:
$w_i$ = number of work cycles on machine $i$

Objective:
$ \max \sum\limits_{i} w_i y_i^*$

Constraints:
$\sum\limits_{i} a_i w_i \leq z$ <-- noise dosage for this schedule

$w_i \leq u_i$ $\forall$ machines $i$

$\sum\limits_{i} t_i w_i \leq H$ <-- time in hours for this schedule

$w_i \geq 0$ $\forall$ machines $i$

If objective value of an optimal solution $\leq 1$, then we do not have any violated constraints in the dual
-- means $y_i^*$ is an optimal solution even with all constraints.

Otherwise, we found a schedule that corresponds to a violated constraint in dual


"""
def subProblem(m, d, S, z, H, t, alpha, u, yi_star):
    """
    m: number of machines
    d: number of work cycles to be executed on each machine
    S: number of schedules
    z: upper bound on noise dosage
    H: upper bound on time in hours
    d: number of work cycles on each machine in each schedule
    t: time in hours for each machine in each schedule
    alpha: noise dosage for each machine in each schedule
    u: upper bound on number of work cycles for each machine
    yi_star: optimal dual solution for the relaxation that only has a subset of the constraints
    """
    model = gp.Model("subProblem")
    model.setParam('OutputFlag', False)
    w = model.addVars(list(range(1, m+1)), vtype=GRB.CONTINUOUS, name="wi")
    model.setObjective(gp.quicksum(w[i] * yi_star for i in range(1, m+1)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(alpha[i-1] * w[i] for i in range(1, m+1)) <= z)
    for i in range(1, m+1):
        model.addConstr(w[i] <= u[i-1])
    model.addConstr(gp.quicksum(t[i-1] * w[i] for i in range(1, m+1)) <= H)
    model.optimize()
    return model.objVal



subProblem(m, d, S, z, H, t, alpha, u, yi_star=6)


"""


1) primal
2) dual with the yi* (.dual
3) new ilp with yi*



scehdule = {  # make as many schedules as you have machines
    1:1,
    2:1,
    3:1,
    ....
}






















let s[i] = number of cycles on machine i

if we want each machine to get at least di cycles, then the constraint is:

sum over s of s[i] >= di for all i


We need to convert this to a maximization problem:

Objective:

max -(sum over s of (xs) )

Constraints:

- sum over s of s[i] <=  -di for all i


Primal:
max -1xs1 - 1xs2 - 1xs3 ... - 1xsS
s.t.
- s1[1] - s2[1] - s3[1] ... - sS[1] <= -d1
- s1[2] - s2[2] - s3[2] ... - sS[2] <= -d2
- s1[3] - s2[3] - s3[3] ... - sS[3] <= -d3
...
- s1[m] - s2[m] - s3[m] ... - sS[m] <= -dm

matrix is m x S

Dual:
min -d1ys1 - d2ys2 - d3ys3... - dmysm
s.t.
- s1[1] - s1[2] - s1[3] ... - s1[m] >= -1
- s2[1] - s2[2] - s2[3] ... - s2[m] >= -1
- s3[1] - s3[2] - s3[3] ... - s3[m] >= -1
...
- sS[1] - sS[2] - sS[3] ... - sS[m] >= -1


matrix is S x m





Now we need to find the dual of this problem:

Objective:

min -sum over i of (di * yi)

Constraints:

- sum over i of s[i] >=




















"""