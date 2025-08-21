# %%
###-------------- Balanced Transportation Models --------------###
#When supply exactly matches demand, the model is balanced.

#Library Importation
import pulp
from pulp import *
#from pulp import GLPK

###--------------Create the LP Problem
prob = LpProblem('Basic_Transportation_Model', LpMinimize)

###--------------Variables with non-negativity integer
# Read the problem carefully. Although I have set it to Integer, that is not necessary.
x11 = LpVariable('Plant_1_City_1', lowBound = 0, cat = 'Integer')
x12 = LpVariable('Plant_1_City_2', lowBound = 0, cat = 'Integer')
x13 = LpVariable('Plant_1_City_3', lowBound = 0, cat = 'Integer')
x14 = LpVariable('Plant_1_City_4', lowBound = 0, cat = 'Integer')
x21 = LpVariable('Plant_2_City_1', lowBound = 0, cat = 'Integer')
x22 = LpVariable('Plant_2_City_2', lowBound = 0, cat = 'Integer')
x23 = LpVariable('Plant_2_City_3', lowBound = 0, cat = 'Integer')
x24 = LpVariable('Plant_2_City_4', lowBound = 0, cat = 'Integer')
x31 = LpVariable('Plant_3_City_1', lowBound = 0, cat = 'Integer')
x32 = LpVariable('Plant_3_City_2', lowBound = 0, cat = 'Integer')
x33 = LpVariable('Plant_3_City_3', lowBound = 0, cat = 'Integer')
x34 = LpVariable('Plant_3_City_4', lowBound = 0, cat = 'Integer')

###--------------Subject To (s.t.) - Add the constraints
p1 = x11 + x12 + x13 + x14
prob += p1 <= 35,"Plant_1"
p2 = x21 + x22 + x23 + x24
prob += p2 <= 50,"Plant_2"
p3 = x31 + x32 + x33 + x34
prob += p3 <= 40,"Plant_3"
c1 = x11 + x21 + x31
prob += c1 >= 45,"City_1"
c2 = x12 + x22 + x32
prob += c2 >= 20,"City_2"
c3 = x13 + x23 + x33
prob += c3 >= 30,"City_3"
c4 = x14 + x24 + x34
prob += c4 >= 30,"City_4"

###--------------Define the Objective Function
z = 8*x11 + 6*x12 + 10*x13 + 9*x14 + 9*x21 + 12*x22 + 13*x23 + 7*x24 + 14*x31 + 9*x32 + 16*x33 + 5*x34
prob += z, "Min_Cost"
#print(prob)

###--------------Display the solution
optimization_result = prob.solve(GLPK (options = ['--ranges', 'sensitivity.txt'])) #The CBC solver will work too
LpStatus[optimization_result]

# Why did we not get a sensitivity.txt file?
# Display the solution
for var in (x11, x12, x13, x14, x21, x22, x23, x24, x31, x32, x33, x34):
    print('Optimal kwh to send down the {} line is: {:1.1f}'.format(var.name,var.value()))
print('-'*50)

for var in (p1, p2, p3):
    print('Optimal number of kwh to produce at {} is: {:1.1f}'.format(var.name,var.value()))
print('-'*50)

for var in (c1, c2, c3, c4):
    print('Electricity supplied to {} is: {:1.1f}'.format(var.name, var.value()))
print('-'*50)

#The Optimized Objective Function value
print(f"Total Cost is: ${value(prob.objective):,.2f}")
# %%
###-------------- Unbalanced Transportation Models --------------###
#If the total supply exceeds the total demand or vice versa, the model is unbalanced.

#Library Importation
import pulp
from pulp import *
#from pulp import GLPK

###--------------Define Nodes
#Supply Nodes (sources) 
Plants = ["1", "2", "3"]
Supply = {"1":35, "2":50, "3":50} #dictionary

#Demand Nodes 
Cities = ["1", "2", "3", "4"]
Demand = {"1":45, "2":20, "3":30, "4":30} #dictionary

###--------------Create a list of cost coefficients
#Each row below represents a plant
Costs = [
        [8,6,10,9], #Plant 1
        [9,12,13,17], #Plant 2
        [14,9,16,5] #Plant 3
        ]

#Make a cost dictionary
costs = makeDict([Plants,Cities], Costs, 0) #makeDict(headers, array, default=None)

###--------------Create the LP Problem
prob = LpProblem('Transportation_Model_with_Dictionaries', LpMinimize)

###--------------Variables
#Create Routes
Routes = [(p,c) for p in Plants for c in Cities]

#Decision Variables
Vars = LpVariable.dicts("Routes", (Plants,Cities), 0, None, LpInteger)

###--------------Define the Objective Function
prob += lpSum([Vars[p][c]*costs[p][c] for (p,c) in Routes]), "Min_Costs"

###--------------Subject To (s.t.) - Add the constraints
#Supply Constraints
for p in Plants:
    prob += lpSum([Vars[p][c] for c in Cities]) <= Supply[p], f"Supply_from_Plant: {p}"

#Demand Constraints
for c in Cities:
    prob += lpSum([Vars[p][c] for p in Plants]) >= Demand[c], f"Demand_to_City: {c}"

###--------------Display the solution
#The problem data is written to an .lp file
prob.writeLP("Power_Transformation_Problem.lp")

#The problem is solved using PuLP's choice of Solver
prob.solve()
print("Status:", LpStatus[prob.status]) #The status of the solution is printed 

#Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

#The Optimized Objective Function Value
print(f"Total Cost of Transportation: ${value(prob.objective):,.2f}")
print('-' * 50)

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)
# %%
