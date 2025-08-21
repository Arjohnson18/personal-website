# %%
###-------------- Minimum Cost Flow Problem --------------###
#Library Importation
import pulp
from pulp import *
import pandas as pd

###--------------Create the LP Problem
prob = LpProblem('Basic_Transportation_Model', LpMinimize)

###--------------Decision Variables
# Read the problem carefully. Although I have set it to Integer, that is not necessary.
x12 = LpVariable('S1_D2', lowBound = 0, cat = 'Integer')
x14 = LpVariable('S1_D4', lowBound = 0, cat = 'Integer')
x23 = LpVariable('S2_D3', lowBound = 0, cat = 'Integer')
x35 = LpVariable('S3_D5', lowBound = 0, cat = 'Integer')
x53 = LpVariable('S5_D3', lowBound = 0, cat = 'Integer')
x54 = LpVariable('S5_D4', lowBound = 0, cat = 'Integer')
x56 = LpVariable('S5_D6', lowBound = 0, cat = 'Integer')
x65 = LpVariable('S6_D5', lowBound = 0, cat = 'Integer')
x74 = LpVariable('S7_D4', lowBound = 0, cat = 'Integer')
x75 = LpVariable('S7_D5', lowBound = 0, cat = 'Integer')
x76 = LpVariable('S7_D6', lowBound = 0, cat = 'Integer')

###--------------Define the Objective Function
z = +30*x12 + 40*x14+ 50*x23 + 35*x35 + 40*x53 + 30*x54 + 35*x56 + 25*x65
+ 50*x74 + 45*x75 + 50*x76
prob += z, "Min_Cost"

###--------------Subject To (s.t.) - Add the constraints
prob += (- x12 - x14) >= -200, 'Node1'
prob += (x12 - x23) >=100, "Node2"
prob += (x23 + x53 - x35) >=60, "Node3"
prob+= (x14 + x54 + x74) >= 80, "Node 4"
prob += (x35 + x65 + x75 - x53 - x54 - x56) >= 170, "Node5"
prob += (x56 + x76 - x65) >= 70, "Node6"
prob += (-x74 - x75 - x76) >= -300, "Node7"

###--------------Display the solution
optimization_result = prob.solve() #Alternately GLPK (options=['--ranges','sensitivity.txt']))
print(f"Status: {LpStatus[optimization_result]}")

for var in (x12, x14, x23, x35, x53, x54, x56, x65, x74, x75, x76):
    print('Optimal number of units to send down the {} line is: {:1.1f}'.format(var.name, var.value()))
print('-' * 50)

#Alternate way to Display the solution
for v in prob.variables():
    if v.varValue == 0:
        print(v.name, "=", v.varValue)
production = [v.varValue for v in prob.variables()]
print('-' * 50)

#Node (Constraint) final value
for var in (node1, node2, node3, node4, node5, node6, node7):
    print('Number of units at {} is: {:1.0f}'.format(var, var.value()))
print('-' * 50)

#Getting the value of the constraint
for constraint in prob.constraints:
    print(prob.constraints[constraint].name, prob.constraints[constraint].value() - prob.constraints[constraint].constant)

#The Optimized Objective Function Value
print(f"Total Cost: ${value(prob.objective):,.2f}")
print('-' * 50)

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)
# %%
###-------------- Traveling Salesman Problem --------------###
#Library Importation
import pulp
from pulp import *
import pandas as pd

###--------------Define Nodes
departure = ['1', '2', '3', '4', '5']
arrival = ['1', '2', '3', '4', '5']

#Distance between Nodes
distance = {
'1': { '1': 0, '2': 44, '3': 35, '4': 13, '5': 19 },
'2': { '1': 47, '2': 0, '3': 23, '4': 29, '5': 50 },
'3': { '1': 31, '2': 24, '3': 0, '4': 12, '5': 26 },
'4': { '1': 16, '2': 27, '3': 15, '4': 0, '5': 18 },
'5': { '1': 23, '2': 44, '3': 25, '4': 21, '5': 0 }
}
###--------------Create the LP Problem
prob = LpProblem("Traveling Salesman Problem", LpMinimize)

###--------------Variables
#Create Routes
routes = [ (d, a) for d in departure for a in arrival ]

#Decision Variables
vars = LpVariable.dicts("arc", (departure, arrival), 0, 1, LpInteger)

###--------------Define the Objective Function
prob += (lpSum([vars[r][c] * distance[r][c] for (r, c) in routes]), # r,c=coordina
"Total distance traveled", )

###--------------Subject To (s.t.) - Add the constraints
#Arrrival Constraint
for a in arrival:
    prob += (lpSum([vars[d][a] for d in departure]) == 1, f"Arrival_{a}", )

#Departure Constraint
for d in departure:
    prob += (lpSum([vars[d][a] for a in arrival]) == 1, f"Departure_{d}", )

#Disallowed arcs
prob += (lpSum([vars[a][a] for a in arrival]) == 0, f"DisAllowedArcs", )

#Flow Constraints
vars_f = LpVariable.dicts("Flow", (departure, arrival), 0, None, LpInteger)
n = len(departure)-1

for d in departure:
    for a in arrival:
        prob += (vars_f[d][a] <= n*vars[d][a] )

for o in ['2','3','4','5']: #non-starting cities
    prob += (lpSum([vars_f[a][o]-vars_f[o][a] for a in arrival]) == 1, f"Flow_balance_{o}", )

###--------------Display the solution
prob.writeLP("TSP.lp")
prob.solve()

print("Status:", LpStatus[prob.status])
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

#The Optimized Objective Function Value
print("Total distance = ", value(prob.objective))
print('-' * 50)

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)
# %%
