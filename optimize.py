import gurobipy as gp
import numpy as np

BUDGET = 3020 
weaponsCost = 150 # limit of 8
offenseCost = 90 # limit of 6
crateCost = 60 # limit of 20
armorCost = 1600 # limit of 1

def bruteForce():
    w = np.array([weaponsCost, offenseCost, crateCost, armorCost])
    solutions = []
    for x1 in range(9):
        for x2 in range(7):
            for x3 in range(21):
                for x4 in range(2):
                    cost = w @ np.array([x1, x2, x3, x4])
                    if cost == BUDGET and x2 == 3:
                        solutions.append( [x1, x2, x3, x4] )
    print('All solutions: ')
    for solution in solutions:
        print(solution, w @ solution)


model = gp.Model("optimize")

"""

objective function: maximize x1 * weaponsCost + x2 * offenseCost + x3 * crateCost + x4 * armorCost

constraints: 

x1 * weaponsCost + x2 * offenseCost + x3 * crateCost + x4 * armorCost <= BUDGET

x1 <= 8
x2 <= 6
x3 <= 20
x4 <= 1

x1, x2, x3, x4 >= 0
"""
# quantities = ['num_weapons', 'num_offense', 'num_crates']
# x = model.addVars(range(len(quantities)), vtype=gp.GRB.INTEGER, name='x')
# costs = [weaponsCost, offenseCost, crateCost]
x1 = model.addVar(vtype=gp.GRB.INTEGER, name="x1_num_weapons")
x2 = model.addVar(vtype=gp.GRB.INTEGER, name="x2_num_offense")
x3 = model.addVar(vtype=gp.GRB.INTEGER, name="x3_num_crates")
x4 = model.addVar(vtype=gp.GRB.INTEGER, name="x4_num_armor")


constraints = [
    x1 <= 8,
    x2 <= 6,
    x3 <= 20,
    x4 <= 1,
    x1 >= 0,
    x2 >= 0,
    x3 >= 0,
    x4 >= 0
]

constraints += [
    #offense items
    x2 == 2,
    x3 == 8,

]


model.addConstrs(c for c in constraints)

z = x1 * weaponsCost + x2 * offenseCost + x3 * crateCost + x4 * armorCost
model.addConstr(z <= BUDGET, "Total cost must be less than budget")

model.setObjective(z, gp.GRB.MAXIMIZE)

model.optimize()

for v in model.getVars():
    print(v.varName, v.x)


# verify that the solution is correct
print(
    f"Total cost: {x1.x * weaponsCost + x2.x * offenseCost + x3.x * crateCost + x4.x * armorCost}"
)

print()
bruteForce()