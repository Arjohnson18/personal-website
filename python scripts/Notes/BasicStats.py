# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:03:19 2024

@author: Arjoh
"""

#Library Importation
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro, normaltest, anderson, expon
import scipy  
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot

#Ignore warnings
warnings.filterwarnings("ignore")

#Import Data
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 12 Data.csv")

###--------------Descriptive Statistics
def analyze_variable(df, column_name):
    stats = {}
    stats['Mean'] = df[column_name].mean()
    stats['Median'] = df[column_name].median()
    stats['Min'] = df[column_name].min()
    stats['Max'] = df[column_name].max()
    stats['Range'] = stats['Max'] - stats['Min']
    stats['Q1'] = df[column_name].quantile(0.25)
    stats['Q2'] = df[column_name].quantile(0.50)
    stats['Q3'] = df[column_name].quantile(0.75)
    stats['IQR'] = stats['Q3'] - stats['Q1']
    stats['95th'] = np.percentile(df[column_name], 95)
    stats['Skew'] = df[column_name].skew()
    stats['StDev'] = df[column_name].std()
    stats['Kurt'] = df[column_name].kurt()

    #Add distribution fitting
    loc, scale = expon.fit(df[column_name])
    stats['Location'] = loc
    stats['Scale'] = scale

    return stats
    
#Defines variables and creates a loop
variables = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
results = []
for var in variables:results.append(analyze_variable(df, var))

#Creates DataFrame using the list of results
AnalysisResults = pd.DataFrame(results, index=variables)
AnalysisResults2 = round(AnalysisResults.T,3)   #Rotating AnalysisResults
print(AnalysisResults2)

#Export Results (optional)
AnalysisResults2.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Assignment 12 DesScr.csv')

###--------------Normality Tests Function
def test_normality(df, column_name):
    normality = {}
    
    #Creates a histogram
    plt.hist(df[column_name])
    plt.title(f'Histogram of {column_name}')
    plt.show()
    
    #Creates a QQ plot (a quartile plot)
    qqplot(df[column_name], line='s')
    plt.title(f'QQ Plot of {column_name}')
    plt.show()

    #Statistical Test 1 - Shapiro-Wilk
    stat_sw, p_sw = shapiro(df[column_name])
    normality['Shapiro-Wilk'] = {'stat': stat_sw, 'p-value': p_sw}
    alpha = 0.05
    if p_sw > alpha:
        sw_result = 'Based on Shapiro-Wilks, the sample looks Gaussian (fail to reject H0)'
    else:
        sw_result = 'Based on Shapiro-Wilks, the sample does not look Gaussian (reject H0)'
    normality['Shapiro-Wilk']['result'] = sw_result

    #Statistical Test 2 – D’Agostino’s K^2
    stat_dak, p_dak = normaltest(df[column_name])
    normality['D’Agostino’s K^2'] = {'stat': stat_dak, 'p-value': p_dak}
    alpha = 0.05
    if p_dak > alpha:
        dak_result = 'Based on D’Agostino’s K^2, the sample looks Gaussian (fail to reject H0)'
    else:
        dak_result = 'Based on D’Agostino’s K^2, the sample does not look Gaussian (reject H0)'
    normality['D’Agostino’s K^2']['result'] = dak_result

    #Statistical Test 3 – Anderson-Darling
    result_ad = anderson(df[column_name])
    normality['Anderson-Darling'] = {'stat': result_ad.statistic, 'critical_values': result_ad.critical_values}
    p = 0
    ad_result = []
    for i in range(len(result_ad.critical_values)):
        sl, cv = result_ad.significance_level[i], result_ad.critical_values[i]
        if result_ad.statistic < cv:
            ad_result.append('%.1f: %.3f, Based on AD, the sample looks Gaussian (fail to reject H0)' % (sl,cv))
        else:
            ad_result.append('%.1f: %.3f, Based on AD, the sample does not look Gaussian (reject H0)' % (sl,cv))
    normality['Anderson-Darling']['result'] = ad_result
    
    #Print the results
    print(f"\nNormality Test Results for {column_name}")
    print(f"Shapiro-Wilk: Statistic = {stat_sw:.3f}, p-value = {p_sw:.3f}")
    print(f"  - Result: {normality['Shapiro-Wilk']['result']}")
    print(f"D’Agostino’s K^2: Statistic = {stat_dak:.3f}, p-value = {p_dak:.3f}")
    print(f"  - Result: {normality['D’Agostino’s K^2']['result']}")
    print(f"Anderson-Darling: Statistic = {result_ad.statistic:.3f}")
    print("Critical Values:")
    for res in ad_result:
        print(f"  - {res}")
    print('------------------------------------------------')

    return normality

#Define variables and run normality tests
variables = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
normality_results = {}

for var in variables: normality_results[var] = test_normality(df, var)

#Convert the results (optional)
normalitytests = pd.DataFrame.from_dict(normality_results, orient='index')
normalitytests.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Normality Test Results.csv')

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

###--------------Probability using CDF Distribution
def test_probability(df, column_name):
    prob = {}
    
    # Calculate the mean and standard deviation of the fitted distribution
    prob['Mean'] = df[column_name].mean()  # Mean
    prob['StDev'] = df[column_name].std()  # Standard deviation (scale parameter for normal distribution)
    
    # 1. Calculate P(X < (mean - 5))
    a = prob['Mean'] - 5
    prob1 = stats.norm.cdf(a,prob['Mean'],prob['StDev'])
    
    # 2. Calculate P(X > (mean + 5))
    b = prob['Mean'] + 5
    prob2 = stats.norm.cdf(b,prob['Mean'],prob['StDev'])
    
    # 3. Calculate P[(mean - (1.5 * s)) < X < (mean + (1.5 * s))]
    lower_bound = prob['Mean'] - (1.5 * prob['StDev'])
    upper_bound = prob['Mean'] + (1.5 * prob['StDev'])
    
    #CDF at the upper bound and lower bound
    prob_lower = stats.norm.cdf(lower_bound, loc=prob['Mean'], scale=prob['StDev'])
    prob_upper = stats.norm.cdf(upper_bound, loc=prob['Mean'], scale=prob['StDev'])
    
    #The probability for the range P[(mean - 1.5 * s) < X < (mean + 1.5 * s)]
    prob3 = prob_upper - prob_lower
    
    #Output the results
    print(f"Probability Test Results for {column_name}")
    print(f"Mean of the Distribution: {prob['Mean']:.3f}")
    print(f"Standard Deviation of the Distribution: {prob['StDev']:.3f}")
    print(f"Lower bound: {lower_bound:.3f}")
    print(f"Upper bound: {upper_bound:.3f}")
    print(f"P(X < Mean - 5): {prob1:.3f}")
    print(f"P(X > Mean + 5): {prob2:.3f}")
    print(f"P(Lower Bound < X < Upper Bound): {prob3:.3f}")
    print('------------------------------------------------')
    
    return prob

#Define variables and run Probability tests
variables = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
ProbResults = []

for var in variables:ProbResults.append(test_probability(df, var))

#Export the results (optional)
print(ProbResults)
prob_results = pd.DataFrame(ProbResults, index=variables)
prob_results.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\ProbResults.csv')


