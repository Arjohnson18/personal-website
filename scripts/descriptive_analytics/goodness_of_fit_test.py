# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:07:01 2024

@author: Arjoh
"""

# Set list of distributions to test
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more
# Turn off code warnings (this is not recommended for routine use)

import warnings
warnings.filterwarnings("ignore")

# Upload Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy  
import scipy.stats
from sklearn.preprocessing import StandardScaler

# Random Test data. Replace this with real data. Turn off to work real data 
np.random.seed(1)
y = np.random.logistic(loc=10, scale=2, size=1000)   #.randn random standard normal sidtribution

#code to import real data for testing. Turn on when ready to do work
#df=pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 10.csv")

# Extract one column to test. Turned off while developing program
#y = df[["column_name_here"]].to_numpy()  #Get variable name from variable explorer


# Create an index array (x) for data
x = np.arange(len(y))
size = len(y)

# view and describe the data
plt.hist(y)
plt.show()
scipy.stats.describe(y)

# Standardize the data. The K-S test assumes data is standardized
sc = StandardScaler()
yy = y.reshape(-1,1)
sc.fit(yy)
y_std = sc.transform(yy)
y_std = y_std.flatten()
del yy

# View and describe the the Standardized data 
plt.hist(y_std)
plt.show()
scipy.stats.describe(y_std)

# Set up list of candidate distributions to use. You can add more here
# See https://docs.scipy.org/doc/scipy/reference/stats.html for more

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

# Set up empty lists to store results in
chi_square = []
p_values = []


# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distributed across all bins 
percentile_bins = np.linspace(0, 100, 51)
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins= percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions
for distribution in dist_names:
    	# Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
    
    	# Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args =param)[1]
        p = np.around(p,5)
        p_values.append(p)

    
    	# Get expected counts in percentile bins
    	# This is based on a 'cumulative distribution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
    
    	# calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency)**2)/ cum_observed_frequency)
        chi_square.append(ss)
        
# Collate results and sort by goodness of fit (best at top)
results = pd.DataFrame()
results['Distribution'] = dist_names
results['Chi-Square'] = chi_square
results['P-Values'] = p_values
results.sort_values(['Chi-Square'], inplace = True)
    
# Report results
print('\nThe null hypothesis for the K-S is H0: Data is from Specified Distribution')   #creates a carriage command
print('Large p-values (P>0.05) mean fail to reject H0')
print('\nDistributions sorted by goodness of fit:')
print('-------------------------------------')
print(results)
results.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\results.csv')