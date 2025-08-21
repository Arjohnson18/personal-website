# %%
#Library Importation
import pulp
import pandas as pd
from pulp import GLPK

###--------------Create the LP Problem
prob = pulp.LpProblem('Ch5_Example', pulp.LpMinimize)

###--------------Decision Variables with non-negativity (sign restriction)
x1= pulp.LpVariable('Monday', lowBound = 0, cat ='Integer')
x2= pulp.LpVariable('Tuesday', lowBound = 0, cat ='Integer')
x3= pulp.LpVariable('Wednesday', lowBound = 0, cat ='Integer')
x4= pulp.LpVariable('Thursday', lowBound = 0, cat ='Integer')
x5= pulp.LpVariable('Friday', lowBound = 0, cat ='Integer')
x6= pulp.LpVariable('Saturday', lowBound = 0, cat ='Integer')
x7= pulp.LpVariable('Sunday', lowBound = 0, cat ='Integer')

###--------------Define the Objective Function
Emp = x1 + x2 + x3 + x4 + x5 + x6 + x7
prob += Emp, "Total Number of Employees"

###--------------Add constraints
prob += (x1 + x4 + x5 + x6 + x7 >= 17), "Monday"
prob += (x1 + x2 + x5 + x6 + x7 >= 13), "Tuesday"
prob += (x1 + x2 + x3 + x6 + x7 >= 15), "Wednesday"
prob += (x1 + x2 + x3 + x4 + x7 >= 19), "Thursday"
prob += (x1 + x2 + x3 + x4 + x5 >= 14), "Friday"
prob += (x2 + x3 + x4 + x5 + x6 >= 16), "Saturday"
prob += (x3 + x4 + x5 + x6 + x7 >= 11), "Sunday"
print(prob)
print('-'*50)

###--------------Solve the LP using the default solver
optimization_result = prob.solve() #using the default CBC solver

###--------------Display the solution
for var in (x1, x2, x3, x4, x5, x6, x7):
    print('Optimal employees for shift {} is: {:1.0f}'.format(var.name, var.value()))

#The Optimized Objective Function value
print("\nTotal number of employees is = ", pulp.value(prob.objective))
print('-'*50)

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
print(sensitivity)

#Print Sensitivity Analysis
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)

###--------------Solve the LP using the GLPK solver
optimization_result2 = prob.solve(GLPK (options=['--ranges', 'Sensitivity_Analysis.txt']))
# %%
