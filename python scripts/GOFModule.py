# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:51:22 2024

@author: Arjoh
"""

#Library Importation
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import StandardScaler

#Ignore warnings
warnings.filterwarnings("ignore")

#Goodness of Fit Test Function
def goodness_of_fit(df, column_name):
    y = df[[column_name]].to_numpy()

    #Standardize data
    sc = StandardScaler()
    y_std = sc.fit_transform(y.reshape(-1, 1)).flatten()

    #Plot standardized data histogram
    plt.hist(y_std)
    plt.title(f'Standardized Data Histogram for {column_name}')
    plt.xlabel('Standardized Values')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    #Distribution names
    dist_names = [
        'alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
        'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib',
        'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 'gennorm', 'genpareto',
        'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'geninvgauss', 'gibrat',
        'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant',
        'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa4', 'kappa3', 'ksone', 'kstwobign',
        'laplace', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke',
        'moyal', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw',
        'powerlognorm', 'powernorm', 'rdist', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 'skewnorm', 't',
        'trapezoid', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line',
        'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'
    ]

    #Initialize results storage
    chi_square = []
    p_values = []

    #Set up 50 bins for chi-square test
    percentile_bins = np.linspace(0, 100, 50)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, _ = np.histogram(y_std, bins=percentile_cutoffs)
    cum_observed_frequency = np.cumsum(observed_frequency)

    #Perform Goodness of Fit Tests
    for distribution in dist_names:
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p_values.append(round(p, 5))

        #Calculate expected counts in bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        expected_frequency = np.diff(cdf_fitted) * len(y_std)
        cum_expected_frequency = np.cumsum(expected_frequency)
        chi_square.append(sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency))

    #Create DataFrame of results and sort
    results_gof = pd.DataFrame({'Distribution': dist_names, 'Chi-Square': chi_square, 'P-Values': p_values})
    results_gof.sort_values('Chi-Square', inplace=True)

    print(f'\nGoodness-of-Fit Test Results for {column_name}')
    print('------------------------------------------------')
    print(round(results_gof, 3))

    return results_gof.set_index('Distribution').T.to_dict()

#Main Function to Run All Tests
def run_gof_tests(df, variables, group_column):
    results = {}
    grouped = df.groupby(group_column) 
    for group_name, group_data in grouped:
        results[group_name] = {var: goodness_of_fit(group_data, var) for var in variables}

    return results