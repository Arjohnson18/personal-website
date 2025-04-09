# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:58:19 2024

@author: Arjoh
"""

#Library Importation
from scipy.stats import binom
from scipy.stats import poisson

#-------------Solving Binomials
# Parameters
n = 10  # Number of trials
p = 0.10  # Probability of success

# Calculate the probability of 3 or more successes
prob = 1 - binom.cdf(2, n, p)
print(f"The probability of 3 or more defective bulbs is {prob:.4f}")

# Given parameters
n = 5  # number of questions
p = 0.2  # probability of getting a correct answer

# Calculate cumulative probability for getting 0, 1, or 2 questions correct
prob_none_correct = binom.cdf(0, n, p)
print(prob_none_correct)

#-------------Solving Poisson Distributions

# Importing the poisson function from scipy.stats for Poisson distribution calculations
# Given parameters
lambda_value = 5  # Average number of events
k = 3  # Number of occurrences

# Calculate the probability P(X = 3) using the Poisson distribution
prob_X_equals_3 = poisson.pmf(k, lambda_value)
print(prob_X_equals_3)
round(prob_X_equals_3,4)
