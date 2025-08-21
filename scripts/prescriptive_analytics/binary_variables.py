# %%
#Library Importation
import pulp
import pandas as pd
from pulp import GLPK

###--------------Create the LP Problem
prob = pulp.LpProblem('A Capital Budget Problem_Relaxation', pulp.LpMaximize)

###--------------Decision Variables with non-negativity (sign restriction)
x1= pulp.LpVariable('Project_1', lowBound = 0, upBound=1)
x2= pulp.LpVariable('Project_2', lowBound = 0, upBound=1)
x3= pulp.LpVariable('Project_3', lowBound = 0, upBound=1)
x4= pulp.LpVariable('Project_4', lowBound = 0, upBound=1)
x5= pulp.LpVariable('Project_5', lowBound = 0, upBound=1)
x6= pulp.LpVariable('Project_6', lowBound = 0, upBound=1)

###--------------Define the Objective Function
z = 141*x1 + 187*x2 + 121*x3 + 283*x4 + 265*x5 + 127*x6
prob += z, "Max_NPU"

###--------------Add constraints
prob += (75*x1 + 90*x2 + 60*x3 + 50*x4 + 10*x5 + 50*x6 <= 275), "Year 1"
prob += (25*x1 + 35*x2 + 15*x3 + 20*x4 + 50*x5 + 20*x6 <= 85), "Year 2"
prob += (10*x1 +  0*x2 + 15*x3 + 15*x4 + 20*x5 + 10*x6 <= 60), "Year 3"
prob += (15*x1 +  0*x2 + 15*x3 + 25*x4 + 20*x5 + 10*x6 <= 60), "Year 4"
prob += (10*x1 + 30*x2 + 15*x3 +  5*x4 + 20*x5 + 10*x6 <= 25), "Year 5"
print(prob)
print('-'*50)

###--------------Solve the LP using the default solver
# We do not really need to check the status. See optimization_result2 in variables: 1=optimal
optimization_result = prob.solve(GLPK (options = ['--ranges', 'sensitivity.txt']))
pulp.LpStatus[optimization_result]

###--------------Display the solution
for var in (x1, x2, x3, x4, x5, x6):
    print('Optimal project selection {} is: {:1.4f}'.format(var.name, var.value()))

#The Optimized Objective Function value
print("Total NPV is = ", pulp.value(prob.objective))
print('-'*50)

# Shadow Prices
o = [{'name':name,'shadow price':c.pi}
for name, c in prob.constraints.items()]
print(pd.DataFrame(o))

###--------------CBC Shadow Price and Slack
o = [{'name':name,'shadow price':c.pi,'slack': c.slack}
for name, c in prob.constraints.items()]
print(pd.DataFrame(o))

#########################################################################
# %%
#Library Importation
import pulp
import pandas as pd
from pulp import GLPK

###--------------Create the LP Problem
prob = pulp.LpProblem('A_Capital_Budget_Problem_Relaxation', pulp.LpMaximize)

###--------------Decision Variables with non-negativity (sign restriction)
x1= pulp.LpVariable('Project_1', cat = 'Binary')
x2= pulp.LpVariable('Project_2', cat = 'Binary')
x3= pulp.LpVariable('Project_3', cat = 'Binary')
x4= pulp.LpVariable('Project_4', cat = 'Binary')
x5= pulp.LpVariable('Project_5', cat = 'Binary')
x6= pulp.LpVariable('Project_6', cat = 'Binary')

###--------------Define the Objective Function
z = 141*x1 + 187*x2 + 121*x3 + 283*x4 + 265*x5 + 127*x6
prob += z, "Max_NPU"

###--------------Add constraints
prob += (75*x1 + 90*x2 + 60*x3 + 50*x4 + 10*x5 + 50*x6 <= 275), "Year 1"
prob += (25*x1 + 35*x2 + 15*x3 + 20*x4 + 50*x5 + 20*x6 <= 85), "Year 2"
prob += (10*x1 +  0*x2 + 15*x3 + 15*x4 + 20*x5 + 10*x6 <= 60), "Year 3"
prob += (15*x1 +  0*x2 + 15*x3 + 25*x4 + 20*x5 + 10*x6 <= 60), "Year 4"
prob += (10*x1 + 30*x2 + 15*x3 +  5*x4 + 20*x5 + 10*x6 <= 25), "Year 5"


print(prob)
print('-'*50)

###--------------Solve the LP using the default solver
# We do not really need to check the status. See optimization_result2 in variables: 1=optimal
optimization_result = prob.solve(GLPK (options = ['--ranges', 'sensitivity.txt']))
pulp.LpStatus[optimization_result]

###--------------Display the solution
for var in (x1, x2, x3, x4, x5, x6):
    print('Optimal project selection {} is: {:1.4f}'.format(var.name, var.value()))

#The Optimized Objective Function value
print("\nTotal NPV is = ", pulp.value(prob.objective))
print('-'*50)

# Shadow Prices
o = [{'name':name,'shadow price':c.pi}
for name, c in prob.constraints.items()]
print(pd.DataFrame(o))

###--------------CBC Shadow Price and Slack
o = [{'name':name,'shadow price':c.pi,'slack': c.slack}
for name, c in prob.constraints.items()]
print(pd.DataFrame(o))
# %%
