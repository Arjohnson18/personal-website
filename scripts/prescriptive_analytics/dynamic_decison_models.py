# %%
#Library Importation
import pulp
import pandas as pd
from pulp import GLPK

###--------------Create the LP Problem
prob = pulp.LpProblem('Ch7_Example', pulp.LpMinimize)

###--------------Variables
#Decision Variables for Production
x1= pulp.LpVariable('January', lowBound = 2000, upBound=4000)
x2= pulp.LpVariable('February', lowBound = 1750, upBound=3500)
x3= pulp.LpVariable('March', lowBound = 2000, upBound=4000)
x4= pulp.LpVariable('April', lowBound = 2250, upBound=4500)
x5= pulp.LpVariable('May', lowBound = 2000, upBound=4000)
x6= pulp.LpVariable('June', lowBound = 1750, upBound=3500)

#Inventory Variables (Ending Inventory each month)
I2 = pulp.LpVariable('Inv_Feb')
I3 = pulp.LpVariable('Inv_Mar')
I4 = pulp.LpVariable('Inv_Apr')
I5 = pulp.LpVariable('Inv_May')
I6 = pulp.LpVariable('Inv_Jun')
I7 = pulp.LpVariable('Inv_End')

###--------------Add constraints
#Production levels
prob += (2000 <= x1 <= 4000), "January"
prob += (1750 <= x2 <= 3500), "February"
prob += (2000 <= x3 <= 4000), "March"
prob += (2250 <= x4 <= 4500), "April"
prob += (2000 <= x5 <= 4000), "May"
prob += (1750 <= x6 <= 3500), "June"

#Beginning Inventory
I1 = 2750
prob += I2 == I1 + x1 - 1000, "Beg_February"
prob += I3 == I2 + x2 - 4500, "Beg_March"
prob += I4 == I3 + x3 - 6000, "Beg_April"
prob += I5 == I4 + x4 - 5500, "Beg_May"
prob += I6 == I5 + x5 - 3500, "Beg_June"
prob += I7 == I6 + x6 - 4000, "Beg_July"

#Ending Inventory = Beg. Invt. + Units Produced - Units Sold
prob += I1 + x1 - 1000 >= 1500, "End_January_Min"
prob += I1 + x1 - 1000 <= 6000, "End_January_Max"

prob += I2 + x2 - 4500 >= 1500, "End_February_Min"
prob += I2 + x2 - 4500 <= 6000, "End_February_Max"

prob += I3 + x3 - 6000 >= 1500, "End_March_Min"
prob += I3 + x3 - 6000 <= 6000, "End_March_Max"

prob += I4 + x4 - 5500 >= 1500, "End_April_Min"
prob += I4 + x4 - 5500 <= 6000, "End_April_Max"

prob += I5 + x5 - 3500 >= 1500, "End_May_Min"
prob += I5 + x5 - 3500 <= 6000, "End_May_Max"

prob += I6 + x6 - 4000 >= 1500, "End_June_Min"
prob += I6 + x6 - 4000 <= 6000, "End_June_Max"

###--------------Define the Objective Function
#Production Cost + Inventory Holding Cost
z = (
    240*x1 + 250*x2 + 265*x3 + 285*x4 + 280*x5 + 260*x6
    + 3.6*(I1 + I2)/2 + 3.75*(I2 + I3)/2 + 3.98*(I3 + I4)/2
    + 4.28*(I4 + I5)/2 + 4.20*(I5 + I6)/2 + 3.9*(I6 + I7)/2)
prob += z, "Production_Inventory_Costs"
#print(prob)
#print('-'*50)

###--------------Display the solution
optimization_result = prob.solve(GLPK (options = ['--ranges', 'Sensitivity_Analysis.txt']))
print(f"Status: {pulp.LpStatus[optimization_result]}")

for var in (x1, x2, x3, x4, x5, x6):
    print('Optimal number of units to build in {}: {:1.0f}'.format(var.name, var.value()))
print('-'*50)

for var in (I2,I3,I4,I5,I6,I7):
    print('The end of period inventory is {:1.0f}' .format(var.value()))

#The Optimized Objective Function value
print(f"Total Cost is: ${pulp.value(prob.objective):,.2f}")
print('-'*50)

###--------------CBC Shadow Price and Slack. Binding constraints have Slack=0
sensitivity = ({'name':name,'shadow price':c.pi, 'slack':c.slack} for name, c in prob.constraints.items())
Sensitivity_Analysis = pd.DataFrame(sensitivity)
print(Sensitivity_Analysis)

# %%
