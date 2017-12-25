#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib
import warnings
warnings.filterwarnings('ignore')

""" 
#Referecnes
-----------
In the dataset analysis and preprocessing phases, I based on the following kernel, on Kaggle :
    https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python    
"""

df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')


''' print the names of df_train's columns  '''
print(df_train.columns)

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,linewidths=0.2);
plt.show()

#descriptive statistics summary of SalePrice, target feature.
print(df_train['SalePrice'].describe())

#Showing the Peakness, Skewness and data distribution
sns.distplot(df_train['SalePrice']);
plt.show()

'''
# Skewness and Kurtosis
-----------------------
I recommend visiting this website to know more about Skewness and Kurtosis:
    https://datascienceconcepts.wordpress.com/2015/11/25/skewness-and-kurtosis-in-nutshell/

We can see clearly in the histograma showed before, how we have positive-skewed situation
in  which theres a normal number of high values. 

Talking about Kurtosis( Peak ) we can see clearly how it s a Leptopkurtic-Platykurtic distribution:
    Leptopkurtic:The peak is high and tail is fat which means the distribution is more clustered around the mean. 
    Platykurtic: The peak is low and tail is thin which means the distribution is less clustered around the mean.
It's a Leptopkurtic-Platykurtic distribution, because the mean is high and the tail is thin.

'''
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

'''
#Relationship with numerical values
-----------------------------------
Plotting a scatter matrix over SalePrice and GrLivArea

In this scatter plot, we can see clearly how it's a positive linear relationship.

To make better understanding of correlation relationships i recommend the next website:
    https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials
     
'''
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()


'''
#Relationship with numerical values
-----------------------------------
Plotting a scatter matrix over SalePrice and TotalBsmtSF

In this scatter plot, we can see clearly how it's a positive linear relationship. However sometimes
TotalBsmtSF give 0 values of for incrementing values of SalePrice. We note how the slope is linear but high.

'''
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()

'''
# Box Plot over overallqual/saleprice
-------------------------------------
We can clearly, see how SalePrice is strongly related with the value of OverallQual feature.

To make better understanding of  of Box Plot:
    https://www.wellbeingatschool.org.nz/information-sheet/understanding-and-interpreting-box-plots
'''

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.show()


'''
# Box Plot over YearBuilt/SalePrice
-------------------------------------

To make better understanding of  of Box Plot:
    In this Box Plot, I could not find  any especial point that could help me to determine the SalePrice value. However
    this feature( YearBuilt ) is well correlated with SalePrice, as we saw in the Heatmap showed before.
    
    But if we see the Scatter plot, we can how recently the SalePrice is being increased, so much faster than old times.

    https://www.wellbeingatschool.org.nz/information-sheet/understanding-and-interpreting-box-plots
'''
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.show()

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()


'''
# Scatter Plot for SalePrice with each feature we considered well correlated and interesting to our data preprocessing.
---------------------------------------------------------------------------------------------------------------------- 
'''
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


'''
# Missing Data ...
------------------
Getting the number of missing data and the perecentage for a feature if contains missing data. 
'''
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

'''
# Dealing with missing data "The simple way"
--------------------------------------------
'''
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...

'''
# Standardizing data
The primary concern here is to establish a threshold that defines an observation as an outlier. To do so,
we'll standardize the data. In this context, data standardization means converting data values to have mean
of 0 and a standard deviation of 1.

I recommend a the following website to make a quick idea about feature standarizing:
    https://www.datasciencecentral.com/profiles/blogs/feature-scaling-and-normalization
'''
df_train_SalePrice = df_train['SalePrice']
saleprice_scaled = StandardScaler().fit_transform(df_train_SalePrice[:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

'''
# Bivariate analysis saleprice/grlivarea
This plot has been showen before, but this time we gonna search for outliars points and delete them if they don't 
belong to the normal group or not following the crowed.

The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is 
happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm 
quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers
and delete them.

The two observations in the top of the plot are those 7.something observations that we said we should be careful about.
They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.

'''
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()


print(df_train[(df_train['GrLivArea']> 4000) & (df_train['SalePrice']> 70000) ][['GrLivArea','SalePrice','Id']] )


'''
#Deleting the outliars points
-----------------------------
After knowing which rows meet the conditions to be deleted, we delete them.
'''
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train = df_train.drop(df_train[df_train['Id'] == 692].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1183].index)
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

'''
#bivariate analysis saleprice/TotalBsmtSF
This plot has been showen before, but this time we gonna search for outliars points and delete them if they don't 
belong to the normal group or not following the crowed.

No rows worth to be deleted
'''
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()

'''
Studying the statistical assumptions of SalePrice ( Normality ) - SalePrice
---------------------------------------------------------------
# Histogram and Normal Probability plot

I recommend the following website to know more about Normality Testing :
    https://en.wikipedia.org/wiki/Normality_test
    
The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:

Histogram - Kurtosis and skewness.
Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

We can see clearly how SalePrice is not normal, as i mentioned before, on Skewness and Kurtosis, SalePrice hase its peak
and show a positive skewness. Then nothing is lost, because with a simple statistical transformation, we can make Sale-
Price more normal. This transformation it's applying the logarithm to SalePrice values, so applying the Logarithm with
positive skewness, works well ! 
    
'''
sns.distplot(df_train['SalePrice'], fit=norm);
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

#Applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

'''
#Transformed histogram and normal probability plot
--------------------------------------------------
We note how we got mor normal values for SalePrice, the mayority of the values are fairly distributed on the 2 sides of
the mean. 
'''
sns.distplot(df_train['SalePrice'], fit=norm);
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()

'''
Studying the statistical assumptions of SalePrice ( Normality ) - GrLivArea
---------------------------------------------------------------
# Histogram and Normal Probability plot

I recommend the following website to know more about Normality Testing :
    https://en.wikipedia.org/wiki/Normality_test

The point here is to test 'GrLivArea' in a very lean way. We'll do this paying attention to:

Histogram - Kurtosis and skewness.
Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

As result, GrLivArea is more normal than SalePrice, but we can make it more normal yet. Using the same techniques used
before with SalePrice.

'''
sns.distplot(df_train['GrLivArea'], fit=norm);
plt.show()
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.show()

#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)


#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1


#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
