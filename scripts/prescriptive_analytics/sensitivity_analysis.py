# %%
#Library Importation
import pulp
import pandas as pd
from pulp import GLPK

###--------------Create the LP Problem
prob = pulp.LpProblem('Ch3_Problem7', pulp.LpMinimize) #or #LpMaximize

###--------------Decision Variables with non-negativity (sign restriction)
x1= pulp.LpVariable('Brownie', lowBound = 0)
x2= pulp.LpVariable('IceCream', lowBound = 0)
x3= pulp.LpVariable('Cola', lowBound = 0)
x4= pulp.LpVariable('Pineapple Cheese Cake', lowBound = 0)

###--------------Define the Objective Function
Cost = 0.5*x1 + 0.2*x2 + 0.3*x3 + 0.8*x4
prob += Cost, "Total Cost of Diet Per Day"

###--------------Add constraints
#Calories
prob += (400 * x1 + 200 * x2 + 150 * x3 + 500 * x4 >= 500), "Calories"
#Chocolate
prob += (3 * x1 + 2 * x2 >= 6), "Chocolate"
#Sugar
prob += (2 * x1 + 2 * x2 + 4 * x3 + 4 * x4 >= 10), "Sugar"
#Fat
prob += (2 * x1 + 4 * x2 + 1 * x3 + 5 * x4 >= 6), "Fat"
print(prob)

###--------------Solve the LP using the default solver
optimization_result = prob.solve() #using the default CBC solver
assert_optimization_result = pulp.LpStatusOptimal #Double check for an optimal solution

###--------------Display the solution
for var in (x1, x2, x3, x4):
    print('Optimal daily quantity of {} to consume: {:1.0f}'.format(var.name, var.varValue))

###--------------The Optimized Objective Function value
print('Total Cost of Daily Diet per Day: {:1.4f}', pulp.value(prob.objective))

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
print(sensitivity)

#Print Sensitivity Analysis
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)

###--------------Solve the LP using the GLPK solver
optimization_result2 = prob.solve(GLPK (options=['--ranges', 'Sensitivity_Analysis.txt']))
# %%
