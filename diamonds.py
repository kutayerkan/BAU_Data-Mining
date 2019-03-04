# -*- coding: utf-8 -*-

"""

https://www.kaggle.com/shivam2503/diamonds

Index: counter
carat: Carat weight of the diamond
cut: Describe cut quality of the diamond. Quality in increasing order Fair, Good, Very Good, Premium, Ideal
color: Color of the diamond, with D being the best and J the worst
clarity: How obvious inclusions are within the diamond:(in order from best to worst, FL = flawless, I3= level 3 inclusions) FL,IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
depth: depth % :The height of a diamond, measured from the culet to the table, divided by its average girdle diameter
table: table%: The width of the diamond's table expressed as a percentage of its average diameter
price: the price of the diamond
x: length mm
y: width mm
z: depth mm
"""

# %% import libraries

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz
import pydot

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

# qt5 for new window, inline for inline
%matplotlib inline

# %% read data, check for missing values

df = pd.read_csv("diamonds.csv", sep=",", index_col=0)
df.head()

print("Number of missing values:", len(df[df.isnull().any(axis=1)]))

# %% create volume feature

df['volume'] = df['x'] * df['y'] * df['z']

pd.set_option('display.max_columns', 12)
df.sample(10)
print(df.head())
print (df.describe())
pd.set_option('display.max_columns', 0)

# %% plot histogram of features

df.hist();

# %% drop volume outlier

print(df.nlargest(10, 'volume'))
df = df.drop(df[df['volume'] > 3000].index)

# %% plot volume & carat, must be ~linear

df.plot.scatter('volume', 'carat');

# %% drop zero depth diamonds

print(df[df['z'] == 0])
df = df.drop(df[df['z'] == 0].index)

# %% drop remaining outliers

print(df[df['volume'] > 800])
df = df.drop(df[df['volume'] > 800].index)

# %% plot volume & carat

df.plot.scatter('volume', 'carat');

# %% plot volume histogram

df['volume'].hist();

# %% bivariate analysis plots

sns.lmplot( x="volume", y="price", data=df, fit_reg=False);

# Cubic transformation after looking at sample data
df['root_price'] = np.cbrt(df['price'])

# %% graphs

sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False);
sns.lmplot( x="carat", y="root_price", data=df, fit_reg=False);
sns.lmplot( x="table", y="root_price", data=df, fit_reg=False);
sns.lmplot( x="depth", y="root_price", data=df, fit_reg=False);

sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False, hue='cut', hue_order=['Ideal','Premium','Very Good','Good','Fair'], palette='RdYlGn_r', scatter_kws={'alpha':0.5}, legend_out=False);
sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False, hue='color', hue_order=['D','E','F','G','H','I','J'], palette='RdYlGn_r',scatter_kws={'alpha':0.5}, legend_out=False);
sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False, hue='clarity', hue_order=['IF','VVS1','VVS2','SI1','SI2','I1'], palette='RdYlGn_r',scatter_kws={'alpha':0.5}, legend_out=False);

sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False, hue='table', palette='RdYlGn', scatter_kws={'alpha':0.5}, legend=False);
sns.lmplot( x="volume", y="root_price", data=df, fit_reg=False, hue='depth', palette='RdYlGn', scatter_kws={'alpha':1}, legend=False);

# %% do ordinal encoding on actual dataset

df['cut'].unique()
df['cut'] = df['cut'].replace({'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal':5})

df['color'].unique()
df['color'] = df['color'].replace({'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F':5, 'E':6, 'D':7})

# FL = flawless, I3= level 3 inclusions) FL,IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
df['clarity'].unique()
df['clarity'] = df['clarity'].replace({'I3': 1, 'I2': 2, 'I1': 3, 'SI2': 4, 'SI1':5, 'VS2':6, 'VS1':7, 'VVS2': 8, 'VVS1':9, 'IF':10, 'FL':11})

# %% plot correlation matrix

corr = df.corr()
sns.heatmap(corr,
            cmap="PiYG",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True);
            
# %% name the label and features

label = df['root_price']
features = df[['volume','clarity','color','cut','table','depth']]
 
# %% do train/test Split

bins = np.linspace(0, len(df), 100)

y_binned = np.digitize(label, bins)

features_train, features_test, label_train, label_test = train_test_split(features,label, test_size=0.25, random_state = 42, shuffle=True, stratify=y_binned)

features_train, features_valid, label_train, label_valid = train_test_split(features_train,label_train, test_size=(1/3), random_state = 42, shuffle=True)            
            
np.mean(label)
np.mean(label_train)
np.mean(label_valid)
np.mean(label_test)            

# %% Decision Tree Regressor - Training Data

#check for min samples and criterion

dt = DecisionTreeRegressor(random_state=22)
dt.fit(features_train, label_train)
label_train_pred = dt.predict(features_train)

print ("\nDecision Tree Performance on Training Data:")
print ("r2: {0:.4f}".format(r2_score(label_train, label_train_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_train, label_train_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_train, label_train_pred)))

# %% Decision Tree - Hyperparameter Tuning

grid={"max_depth":[1,2,3,4,5,6,7,8,9,10,11,12],
      "min_samples_split":[10,20,50,100],
      "min_samples_leaf":[1,5,10,20,50,100]}

dt_cv=GridSearchCV(dt,grid, n_jobs=-1) # use all processors
dt_cv.fit(features_valid,label_valid)

print("Tuned Hyperparameters:\n",dt_cv.best_params_)

# %% Decision Tree Regressor - Validation Data

dt = DecisionTreeRegressor(random_state=22)
dt.fit(features_train, label_train)
label_valid_pred = dt.predict(features_valid)

print ("\nDecision Tree Performance on Validation Data, NO Hyperparameter Tuning:")
print ("r2: {0:.4f}".format(r2_score(label_valid, label_valid_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_valid, label_valid_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_valid, label_valid_pred)))

dt = DecisionTreeRegressor(random_state=22, criterion='mse', max_depth=11, min_samples_split=20, min_samples_leaf=5)
dt.fit(features_train, label_train)
label_valid_pred = dt.predict(features_valid)

print ("\nDecision Tree Performance on Validation Data, WITH Hyperparameter Tuning:")
print ("r2: {0:.4f}".format(r2_score(label_valid, label_valid_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_valid, label_valid_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_valid, label_valid_pred)))

# %% Linear Regression

reg = LinearRegression()
reg.fit(features_train, label_train)
label_train_pred = reg.predict(features_train)

print('Warning: MSLE throws error when label = price (not root_price)')

print ("\nLinear Regression Performance on Training Data:")
print ("r2: {0:.4f}".format(r2_score(label_train, label_train_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_train, label_train_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_train, label_train_pred)))

label_valid_pred = reg.predict(features_valid)

print ("\nLinear Regression Performance on Validation Data:")
print ("r2: {0:.4f}".format(r2_score(label_valid, label_valid_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_valid, label_valid_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_valid, label_valid_pred)))

print("\nIntercept:", reg.intercept_)
print("Coefficients:\n", list(zip(list(features_train), reg.coef_)))

# %% Final Performance

dt = DecisionTreeRegressor(random_state=22, criterion='mse', max_depth=11, min_samples_split=20, min_samples_leaf=5)
dt.fit(features_train, label_train)
label_test_pred = dt.predict(features_test)

print ("\nDecision Tree Performance on Test Data")
print ("r2: {0:.4f}".format(r2_score(label_test, label_test_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_test, label_test_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_test, label_test_pred)))

reg = LinearRegression()
reg.fit(features_train, label_train)
label_test_pred = reg.predict(features_test)

print ("\nLinear Regression Performance on Test Data:")
print ("r2: {0:.4f}".format(r2_score(label_test, label_test_pred)))
print ("Mean Absolute Error: {0:.4f}".format(mean_absolute_error(label_test, label_test_pred)))
print ("Mean Squared Log Error: {0:.4f}".format(mean_squared_log_error(label_test, label_test_pred)))

print("\nIntercept:", reg.intercept_)
print("Coefficients:\n", list(zip(list(features_train), reg.coef_)))

# %% write decision tree to image

outfile = tree.export_graphviz(dt, out_file='tree.dot', feature_names=list(features_train))
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
graph.write_pdf('tree.pdf')












