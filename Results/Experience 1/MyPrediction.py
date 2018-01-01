
#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "Data"]).decode("utf8")) #check the files available in the directory

""" ¡¡¡¡ Let's Start !!!"""

# Importing our data set (train and test)
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

# Getting the train columns's names
print("Train data set index :\n " + str(train.index) + "\n" )

# Getting the test columns's names
print(test.head(5))

# Getting the number if rows and features in each data set.
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))


# Making a copy of the 'Id' column of each data set
train_ID = train['Id']
test_ID = test['Id']

# Drop the  'Id' column. Will only be used in the submession file.
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

# Getting the number of rows and features in each data set, after the Id column's been droped.
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))


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

print("\nSalePrice Skewness before being transformed: %f" % train['SalePrice'].skew())
print("SalePrice Kurtosis before being transformed: %f" % train['SalePrice'].kurt())

'''
Fitting train['SalePrice'] and Plotting its distribution before and after applyig Log transformation
'''
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


#Applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])

'''
#Transformed histogram and normal probability plot
--------------------------------------------------
We note how we got mor normal values for SalePrice, the mayority of the values are fairly distributed on the 2 sides of
the mean. 
'''
#Check the new distribution
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.show()

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

print("\nSalePrice Skewness after being transformed with log: %f" % train['SalePrice'].skew())
print("SalePrice Kurtosis after being transformed with log: %f" % train['SalePrice'].kurt())


#Obtaining the rows number of each data set, train and test
train_rows_number = train.shape[0]
test_rows_number = test.shape[0]

# Save the Target Column values. So it will be dropped in the following steps.
y_train = train.SalePrice.values

#Concatenating the train and test data sets
Global_Data = pd.concat((train, test)).reset_index(drop=True)

# Getting the new data shape
print("Global_Data size is : {}".format(Global_Data.shape))
Global_Data_Cardinality= Global_Data.shape[0] #the rows number of Global Data

# Detecting duplicates ... In this step we delete the duplicated rows
Global_Data.drop_duplicates()
Global_Data_rows_without_duplicates = Global_Data.shape[0]
duplicates_number = Global_Data_Cardinality - Global_Data_rows_without_duplicates
print("\nThere are "+str(duplicates_number)+" DUPLICATES in Global_Data\n")


# Drop the Targer column ...
Global_Data.drop(['SalePrice'], axis=1, inplace=True)
print("Global_Data size after deleting the Targer column is : {}".format(Global_Data.shape))


"""
Detecting Anamolies (noise) ... Afrer the Target column 'SalePrice' has been droped, and the duplicates has been 
deleted, If now, we return to search for duplicates, that means those are not duplicates but noise, cause they have
the same values for all features except the Target column.
"""
Global_Data.drop_duplicates()
noise_number = Global_Data_Cardinality - Global_Data.shape[0]
print("\nThere are "+str(noise_number)+" NOISE in Global_Data\n")

"""
As we can see we're so lucky to not have any duplicates or noise data, but let's see if we have no missing data too. 
¡¡ I Hope So !!
"""

"""
# Missing Data Treatment ...
In these following steps we gonna analyze the missing data and see how we gonna treat them in out data set.
"""
Global_Data_NA = (Global_Data.isnull().sum() / len(Global_Data)) * 100
Global_Data_NA = Global_Data_NA.drop(Global_Data_NA[Global_Data_NA == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :Global_Data_NA})
print("Missing Ratio is rated by % values:\n"+str(missing_data.head( int(Global_Data.shape[1] ) ) ) )


# Representing the missing data percentage in each feature
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=Global_Data_NA.index, y=Global_Data_NA)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

"""
We can sea clearly how some features, almost has nothing of values ..
 
I think dropping the features with more than 90% of missing values, "probably" will have no effect on our prediction re-
ults. In this case we have :
    PoolQC               99.657
    MiscFeature          96.403
    Alley                93.217
    
"I will study both cases, cause nothing is certain in data science :)". 
"""
print("feature with more than 90% of missing values: "+(missing_data[missing_data['Missing Ratio'] >= 90 ]).index)
Global_Data = Global_Data.drop((missing_data[missing_data['Missing Ratio'] >= 90 ]).index,1)

"""
Let's try to fill the missing data of each feature with the most appropriate values, as we think that so it is ...
"""
# MSZoning NA in pred. filling with most popular values
Global_Data['MSZoning'] = Global_Data['MSZoning'].fillna(Global_Data['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
Global_Data['LotFrontage'] = Global_Data['LotFrontage'].fillna(Global_Data['LotFrontage'].mean())

# Fence Na in all. NA means no fence
Global_Data["Fence"] = Global_Data["Fence"].fillna("No Fence")


# Converting OverallCond to str
Global_Data.OverallCond = Global_Data.OverallCond.astype(str)

# MasVnrType, MasVnrArea NA in all. filling with most popular values
Global_Data['MasVnrType'] = Global_Data['MasVnrType'].fillna(Global_Data['MasVnrType'].mode()[0])
Global_Data["MasVnrArea"] = Global_Data["MasVnrArea"].fillna(Global_Data['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    Global_Data[col] = Global_Data[col].fillna('NoBSMT')

"""
BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for
having no basement.
"""
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    Global_Data[col] = Global_Data[col].fillna(0)

# TotalBsmtSF  NA in pred. I suppose NA means 0
Global_Data['TotalBsmtSF'] = Global_Data['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
Global_Data['Electrical'] = Global_Data['Electrical'].fillna(Global_Data['Electrical'].mode()[0])

# KitchenAbvGr to categorical
Global_Data['KitchenAbvGr'] = Global_Data['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
Global_Data['KitchenQual'] = Global_Data['KitchenQual'].fillna(Global_Data['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
Global_Data['FireplaceQu'] = Global_Data['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    Global_Data[col] = Global_Data[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    Global_Data[col] = Global_Data[col].fillna(0)

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    Global_Data[col] = Global_Data[col].fillna('None')

# SaleType NA in pred. filling with most popular values
Global_Data['SaleType'] = Global_Data['SaleType'].fillna(Global_Data['SaleType'].mode()[0])

"""
LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other ho-
uses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
"""
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
Global_Data["LotFrontage"] = Global_Data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

"""
Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house 
with 'NoSewa' is in the training set, this feature won't help in predictive modeling. We can then safely remove it.
"""
Global_Data = Global_Data.drop(['Utilities'], axis=1)

#Functional : data description says NA means typical
Global_Data["Functional"] = Global_Data["Functional"].fillna("Typ")

"""
Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the 
most common string
"""
Global_Data['Exterior1st'] = Global_Data['Exterior1st'].fillna(Global_Data['Exterior1st'].mode()[0])
Global_Data['Exterior2nd'] = Global_Data['Exterior2nd'].fillna(Global_Data['Exterior2nd'].mode()[0])

#SaleType : Fill in again with most frequent which is "WD"
Global_Data['SaleType'] = Global_Data['SaleType'].fillna(Global_Data['SaleType'].mode()[0])

#MSSubClass : Na most likely means No building class. We can replace missing values with None
Global_Data['MSSubClass'] = Global_Data['MSSubClass'].fillna("None")

"""
Now let's check if we still have missing values ..
"""
#Check remaining missing values if any
Global_Data_NA = (Global_Data.isnull().sum() / len(Global_Data)) * 100
Global_Data_NA = Global_Data_NA.drop(Global_Data_NA[Global_Data_NA == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :Global_Data_NA})
print("Missing Ratio is rated by % values:\n"+str(missing_data.head( int(Global_Data.shape[1] ) ) ) )

"""
As we can see, there's no more missing values in our data set ...
"""

"""
# Transform a numerical feature to categorical, if it's appropiate ..

Sometimes, a feature is represented as numerical, but it in reality it can be treated as categorical, like a year feature
. A year feature can be considered as a categorical feature with 12 categories. 
"""
#MSSubClass = The building class
Global_Data['MSSubClass'] = Global_Data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
Global_Data['OverallCond'] = Global_Data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
Global_Data['YrSold'] = Global_Data['YrSold'].astype(str)
Global_Data['MoSold'] = Global_Data['MoSold'].astype(str)

"""
Let's have a Far-View on, before Modelling. In this section I'll argument about the next points:
    * Skewed Lables
    * Label's Hot Encoding
"""
#Obtaining the numerical features ..
numeric_feats = Global_Data.dtypes[Global_Data.dtypes != "object"].index

# Checking the skewness of all numerical features
skewed_feats = Global_Data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
print("Let's make a look on the skewness of our lables:\n"+str(skewness.head(Global_Data.shape[1])))

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to be transformed by numpy log+1 transformation".format(skewness.shape[0]))


skewed_features = skewness.index
for feat in skewed_features:
    Global_Data[feat] = np.log1p(Global_Data[feat])

# Checking the skewness of all numerical features after being transformed
skewed_feats = Global_Data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
print("Let's make a look on the skewness of our lables after being transformed:\n"+str(skewness.head(Global_Data.shape[1])))


"""
# Labels Hot Encoding ..
-----------------------

Now we're done with the skewness section, let's look on our categorical features ..

¿ Why Labels Hot Encoding ?
    * Hot Encoding means, transform the categorical features into many binary features as much as categories has the feature.
    
    * We apply this technique because some interesting and powerfull alogrithms, don't accept non-numerical values, like
      XGboost for example. So we don't want to lose such an interesting experiece like applying XGboost in our Modeling
      process.    
    
I recommend the following website to make better understanding of Labels Hot Encoding:
    https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science
"""
Global_Data = pd.get_dummies(Global_Data)
print("Our data set shape after applying Hot Encoding on our categorical features:\n" + str(Global_Data.shape) )



"""
Obtaining the new train and test data set ...

Now after finishing our data preprocessing, we can get the new train and test data sets, and then start modeling.
"""
train = Global_Data[:train_rows_number]
test = Global_Data[train_rows_number:]



# Now let's start modeling ...
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb


#Define a cross validation strategy
n_folds = 5 # number of iterations of cross validation.

def rmsle_cv(model):
    """
    * In 'rmsle_cv' we use the cross_val_score function of Sklearn. However this function has not a shuffle attribut,
      we add then one line of code, in order to shuffle the dataset prior to cross-validation.

    :param model: the model to be trained by cross validation technique
    :return rmse: the Root Mean Squared Error

    To know more about RMSE :
        https://gerardnico.com/wiki/data_mining/rmse
    """
    kf = KFold(n_folds, shuffle=True, random_state=123456).get_n_splits(train.values) # Shuffeling the training data set
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

# Let's design some models ..

"""
Gradient Boosting Regression :
    With huber loss that makes it robust to outliers
"""
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

#XGBoost model
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,nthread = -1)
#LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#Elastic Net
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

"""
Lasso:

This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's 
Robustscaler() method on pipeline
"""
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# Let's score our models ...
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


def rmsle(y, y_pred):
    """

    :param y: The Target column values.
    :param y_pred: Trained prediction model
    :return:
    """
    return np.sqrt(mean_squared_error(y, y_pred)),r2_score(y, y_pred)



# Trainging and predicting with XGBoost model.
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print("XGBoost: "+str(rmsle(y_train, xgb_train_pred)))

# Trainging and predicting with LightGBM model.
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print("LGB: "+str(rmsle(y_train, lgb_train_pred)))

# Trainging and predicting with Elastic Net model.
ENet.fit(train, y_train)
ENet_train_pred = ENet.predict(train)
ENet_pred = np.expm1(ENet.predict(test.values))
print("Elastic Net: "+str(rmsle(y_train, ENet_train_pred)))

# Trainging and predicting with Lasso model.
lasso.fit(train, y_train)
lasso_train_pred = lasso.predict(train)
lasso_pred = np.expm1(lasso.predict(test.values))
print("Lasso: "+str(rmsle(y_train, lasso_train_pred)))

# Trainging and predicting with GBoost model.
GBoost.fit(train, y_train)
GBoost_train_pred = GBoost.predict(train)
GBoost_pred = np.expm1(GBoost.predict(test.values))
print("GBoost: "+str(rmsle(y_train, GBoost_train_pred)))


# Ensembling result ...
Ensemble_Result = xgb_pred*0.20 + lgb_pred*0.20 + ENet_pred*0.60

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = Ensemble_Result
sub.to_csv('submission.csv',index=False)
