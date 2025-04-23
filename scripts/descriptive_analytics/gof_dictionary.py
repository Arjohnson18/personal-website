# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:39:36 2024

@author: Arjoh
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from sklearn.preprocessing import StandardScaler

#Ignore warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 12 Data.csv")

###--------------Goodness of Fit Tests
def goodness_of_fit(df, column_name):
    y = df[[column_name]].to_numpy()

    #Create an index array (x) for data
    x = np.arange(len(y))
    size = len(y)
    
    #Standardize the data (K-S test assumes standardized data)
    sc = StandardScaler()
    yy = y.reshape(-1,1)
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()

    #View and describe the standardized data
    plt.hist(y_std)
    plt.title(f'Standardized Data Histogram for {column_name}')
    plt.show()
    
    #Distribution names
    dist_names = ['alpha',
            'anglit',
            'arcsine',
            'argus',
            'beta',
            'betaprime',
            'bradford',
            'burr',
            'burr12',
            'cauchy',
            'chi',
            'chi2',
            'cosine',
            'crystalball',
            'dgamma',
            'dweibull',
            'erlang',
            'expon',
            'exponnorm',
            'exponweib',
            'exponpow',
            'f',
            'fatiguelife',
            'fisk',
            'foldcauchy',
            'foldnorm',
            'genlogistic',
            'gennorm',
            'genpareto',
            'genexpon',
            'genextreme',
            'gausshyper',
            'gamma',
            'gengamma',
            'genhalflogistic',
            'geninvgauss',
            'gibrat',
            'gompertz',
            'gumbel_r',
            'gumbel_l',
            'halfcauchy',
            'halflogistic',
            'halfnorm',
            'halfgennorm',
            'hypsecant',
            'invgamma',
            'invgauss',
            'invweibull',
            'johnsonsb',
            'johnsonsu',
            'kappa4',
            'kappa3',
            'ksone',
            'kstwobign',
            'laplace',
            'levy',
            'levy_l',
            'logistic',
            'loggamma',
            'loglaplace',
            'lognorm',
            #'loguniform',
            'lomax',
            'maxwell',
            'mielke',
            'moyal',
            'nakagami',
            'ncx2',
            'ncf',
            'nct',
            'norm',
            'norminvgauss',
            'pareto',
            'pearson3',
            'powerlaw',
            'powerlognorm',
            'powernorm',
            'rdist',
            'rayleigh',
            'rice',
            'recipinvgauss',
            'semicircular',
            'skewnorm',
            't',
            'trapezoid',
            'triang',
            'truncexpon',
            'truncnorm',
            'tukeylambda',
            'uniform',
            'vonmises',
            'vonmises_line',
            'wald',
            'weibull_min',
            'weibull_max',
            'wrapcauchy']

    #Set up empty lists to store results in
    chi_square = []
    p_values = []
    
    #Set up 50 bins for the chi-square test
    percentile_bins = np.linspace(0, 100, 50)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = np.histogram(y_std, bins=percentile_cutoffs)
    cum_observed_frequency = np.cumsum(observed_frequency)
    
    #Loop through candidate distributions
    for distribution in dist_names:
        #Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        #Obtain the K-S test p-value, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)

        #Get expected counts in percentile bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        #Calculate chi-square
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency)**2) / cum_observed_frequency)
        chi_square.append(ss)
    
    #Collate results and sort by goodness of fit (best at top)
    results_gof = pd.DataFrame({
        'Distribution': dist_names,
        'Chi-Square': chi_square,
        'P-Values': p_values})
    results_gof.sort_values(['Chi-Square'], inplace=True)
    
    #Print results
    print(f'\nGoodness-of-Fit Test Results for {column_name}')
    print('------------------------------------------------')
    print(round(results_gof,3))
    
    #Create a dictionary for results
    results_dict = results_gof.set_index('Distribution').T.to_dict()
    
    return results_dict

#Define variables and run GOF tests
variables = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
GOFResults = {}

for var in variables: GOFResults[var] = goodness_of_fit(df, var)

#Export the results (optional)
gof_results = pd.DataFrame.from_dict(GOFResults, orient='index')
gof_results.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\GOFResults.csv')



