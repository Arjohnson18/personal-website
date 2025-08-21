# %%
#Library Importation
import pulp

###---------------- Minimize Problem -----------------###
###--------------Create the LP Problem
prob = pulp.LpProblem('Ch3_Problem7', pulp.LpMinimize) #or #LpMaximize

###--------------Decision Variables with non-negativity (sign restriction)
#Basic format: pulp.LpVariable(name, lowBound=None, upBound=None, cat='Continuous', e=None
    #Name: the name of the variable
    #lowBound: The lower boundary of this variable's range; default is negative infinity; establishes non-negativity
    #upBound: The upper boundary of this variable's range; default is positive infinity
    #cat: The category this variable is in, 'Integer', 'Binary', or 'Continuous' (default)
    #e: Used for column-based modeling: relates to the variable's existence in the objective function and constraints
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

###--------------The optimized Objective Function value
print('Total Cost of Daily Diet per Day =', pulp.value(prob.objective))

###---------------- Maximize Problem -----------------###
###--------------Create the LP Problem
prob = pulp.LpProblem('Texas Red Soda Problem', pulp.LpMaximize)

###--------------Decision Variables with non-negativity (sign restriction)
x1 = pulp.LpVariable('Social Media', lowBound=0, cat = 'Integer')
x2 = pulp.LpVariable('Television', lowBound=0, cat = 'Integer')

###--------------Define the Objective Function
Z = 36000*x1 + 22500*x2
prob += Z, "Expected Profit"

###--------------Add constraints
prob += (2000*x1 +5000*x2 <= 100000), "Funding"
print(prob)

###--------------Solve the LP using the default solver
optimization_result = prob.solve()
assert_optimization_result = pulp.LpStatusOptimal 

###--------------Display the solution
for var in (x1, x2):
    print('Optimal quantity of {} advertising is: {:1.2f}'.format(var.name, var.varValue))

###--------------The optimized Objective Function value
print("Expected profit from advertising = ", pulp.value(prob.objective))

# %%
