#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# get_ipython().run_line_magic('matplotlib', 'inline')

from fancyimpute import KNN
from scipy.stats import chi2_contingency

os.chdir("C:\\Users\MOULIESWARAN\Desktop\DS py code")
os.getcwd()


# In[46]:


# Import the data
train = pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")

train.shape


# In[47]:


print(train.info())

print(train.describe())
train.head()

# First, let's split the datetime field 'pickup_datetime' to the following -

# year
# month
# date
# hour
# day of week
# In[48]:


# Data cleaning
def prepare_time_features(df):
    df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'],errors='coerce')
    df['Year']= df['pickup_datetime'].apply(lambda x: x.year)
    df['Month']= df['pickup_datetime'].apply(lambda x: x.month)
    df['Date']= df['pickup_datetime'].apply(lambda x: x.day)
    df['Day_of_Week'] = df['pickup_datetime'].apply(lambda x: x.dayofweek)
    df['Hour']= df['pickup_datetime'].apply(lambda x: x.hour)  
    return df


train = prepare_time_features(train)
test = prepare_time_features(test)
train.head()


# In[49]:


#passenger data can't be decimal so make it to floor value
# In given dataset, there are 57 data with 0 passenger count and remove it
# max 6 numbers allowed in a cab, Remove the data with passenger count greater than 7 


def prepare_passenger_count(df):
    df['passenger_count'] = df['passenger_count'].apply( lambda x: np.floor(x))
    df = df.drop( (df[ (df['passenger_count'] > 6) | (df['passenger_count'] < 1)]).index, axis=0)
    return df
 

train = prepare_passenger_count(train)
test =prepare_passenger_count(test)
# train['passenger_count'].value_counts()


# In[50]:



train['fare_amount'] = pd.to_numeric(train['fare_amount'],'coerce')
train['fare_amount'].sort_values()

# Fare amount cannot be negative,0 and 0.01

def prepare_fare_amount(df):
    df = df.drop( (df[ df['fare_amount'] < 1]).index, axis=0)
    return df

train['fare_amount'] = prepare_fare_amount(train)
train['fare_amount'] = pd.to_numeric(train['fare_amount'],'coerce')

# train['fare_amount'].sort_values()
train.dtypes


# In[51]:


# As per train describe, Latitudes range from -90 to 90. Longitudes range from -180 to 180. The above describe clearly shows some outliers. Let's filter them
def prepare_lat_lon(df):   
    df = df.drop( (df[df['pickup_latitude'] < -90] | df[df['pickup_latitude'] > 90]).index)
    df = df.drop((df[df['dropoff_latitude'] < -90] | df[df['dropoff_latitude'] > 90]).index)
    df = df.drop((df[df['pickup_longitude'] < -180] | df[df['pickup_longitude'] > 180]).index)
    df = df.drop((df[df['dropoff_longitude'] < -180] | df[df['dropoff_longitude'] > 90]).index)
    return df
    
train = prepare_lat_lon(train)
test = prepare_lat_lon(test)
train.shape


# ### *MISSING VALUE ANALYSIS*

# In[52]:


missing_val1 = pd.DataFrame(train.isnull().sum().reset_index().rename(columns={'index':'Variables',0:'MissingCount'}))
missing_val1
train = train.dropna()
train.shape
train.isnull().sum()

create a new field 'distance' to fetch the distance between the pickup and the drop.

We can calulate the distance in a sphere when latitudes and longitudes are given by Haversine formula

haversine(θ) = sin²(θ/2)

Calculate the distance based on longitude and latitude

Haversine formula:
dlon = lon2 - lon1
dlat = lat2 - lat1
a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2
c = 2 * atan2( sqrt(a), sqrt(1-a) )
d = R * c (where R is the radius of the Earth)
Eventually in other expression, the formual boils down to the following where φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km) to include latitude and longitude coordinates (A and B in this case).

a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)

c = 2 * atan2( √a, √(1−a) )

d = R ⋅ c

d = Haversine distance
# In[53]:


from math import radians, cos, sin, asin, sqrt
def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    d = 6371* c
    return d


train['distance']=train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)

test['distance']=test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[54]:


# there are 457 records with 0 distance in train data and remove it
train.drop( train[ train['distance'] == 0].index, inplace = True  )

test.drop( test[ test['distance'] == 0].index, inplace = True  )


# In[55]:


# Better graphical view for Exploratory data analysis

#Removing the distance which are > 150 
train.drop( train[ train['distance'] > 150].index, inplace = True  )

#Removing the fare_amount >1000
train.drop( train[ train['fare_amount'] > 1000].index, inplace = True  )


# ### *Exploratory Data Analysis*

# In[57]:


#Target variable distribution

plt.figure(figsize=(15,7))
sns.distplot(train['fare_amount'],kde=True, bins = 40,color='b')
plt.show()


print("Skewness: %f" % train['fare_amount'].skew())
print("Kurtosis: %f" % train['fare_amount'].kurt())


# **1. Does the number of passengers affect the fare?**

# In[58]:


plt.figure(figsize=(15,7))
# plt.hist(train['passenger_count'], bins=15)
sns.countplot(train['passenger_count'])
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')
plt.show()


# In[59]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.show()


# From the above 2 graphs we can see that single passengers are the most frequent travellers,
# and the highest fare also seems to come from cabs which carry just 1 passenger.

#  **2. Does the date and time of pickup affect the fare?**

# In[60]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.show()


# The fares through out the month mostly seem uniform, with the maximum fare received on the 3th

# In[61]:


plt.figure(figsize=(15,7))
# plt.hist(train['Hour'], bins=100)
sns.countplot(train["Hour"])
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.show()


# Interesting! The time of day definitely plays an important role. 
# The frequency of cab rides seem to be the lowest at 5AM and the highest at 6PM.

# In[62]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.show()


# The highest fare receive at 7 am and 10pm.  Other than, fare rate seems to high between 7am to 10am

# **3. Does the day of the week affect the fare?**

# In[63]:


plt.figure(figsize=(15,7))
# plt.hist(train['Day of Week'], bins=100)
sns.countplot(train["Day_of_Week"])
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.show()


# Day of the week doesn't seem to have that much of an influence on the number of cab rides. 
# Mostly saturday have maximum number of ride

# In[64]:


plt.figure(figsize=(15,7))
plt.scatter(x=train['Day_of_Week'], y=train['fare_amount'], s=1.5)
plt.xlabel('Day of Week')
plt.ylabel('Fare')
plt.show()


# The fares through out the Day of week mostly seem uniform, with the maximum fare received on friday and saturday

# In[65]:


def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize = [14,10])
    for i,x in enumerate(timeframes):
        df.loc[:,[x,value]].groupby([x]).mean().plot(ax=ax[i],color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title("{} by {}".format(value.replace("_", " ").title(), x.replace("_", " ").title()))
        ax[i].set_xlabel("")
    ax[len(timeframes)-1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    plt.show()
    
time_slicer(train, ['Year',"Day_of_Week", "Month", "Date", "Hour",'distance'], "fare_amount", "blue")


# ### Outlier analysis

# In[66]:


# Before removing Outlier 
plt.figure(figsize=(15,7))
sns.boxplot(train['distance'],orient= "v")
plt.show()


# In[67]:



plt.figure(figsize=(15,7))
sns.boxplot(train['fare_amount'],orient= "v")
plt.show()


# In[68]:


df = train.copy()
cnames = ['distance','fare_amount']

for i in cnames:
    q75,q25 = np.percentile(df.loc[:,i],[75,25])
    iqr = q75 - q25
    
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr *1.5)
    
    train = train.drop( train[train.loc[:, i] <minimum ] .index)    
    train = train.drop( train[ train.loc[:,i] > maximum].index)


# In[69]:


# After removing outliers
plt.figure(figsize=(15,7))
sns.boxplot(train['distance'],orient= "v")
plt.show()


# In[70]:



plt.figure(figsize=(15,7))
sns.boxplot(train['fare_amount'],orient= "v")
plt.show()


# In[71]:


def convert_data_types(df):

    df['Year'] = df['Year'].astype('object')
    df['Month'] = df['Month'].astype('object')
    df['Date'] = df['Date'].astype('object')
    df['Day_of_Week'] = df['Day_of_Week'].astype('object')
    df['Hour'] = df['Hour'].astype('object')
    return df
    
train = convert_data_types(train)
test = convert_data_types(test)


# In[72]:


numeric_names = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance','fare_amount']
df_corr = train.loc[:,numeric_names]
f,ax = plt.subplots( figsize = (12,7))
sns.heatmap( df_corr.corr(),annot= True )
plt.show()


# ### Feature scaling

# In[73]:


def feature_distance(data):
    return ( data['distance'] - min(data['distance']) ) / (max(data['distance']) - min( data['distance']))

train['distance']= feature_distance(train)
test['distance']= feature_distance(test)


# ### MODELLING AND PREDICTION

# In[74]:


#diividing  Test and train data  using skilearn   train_test_split 

train_feature_selection = train.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis = 1)
# df_feature_selection.shape
# train_feature_selection.to_csv("cab_final.csv",index = False)
train_feature_selection.iloc[:,1:].head()


# In[75]:


from sklearn.model_selection import train_test_split,GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(train_feature_selection.iloc[:,1:],train_feature_selection['fare_amount'],
                                                    test_size=0.4)


# **Evaluation Metric**

# In[76]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score

def RMSE(y_actual, y_predict, Train = True):
    
    mape = np.mean(np.abs((y_actual - y_predict) / y_predict))*100
    if Train:
        print(" Trained Model performance:  ")
  
    else:
        print(" Predicted Model performance:  ")
    print(" MAPE: \t",mape)
#         print(" MSE: \t",mean_squared_error(y_actual, y_predict))    
    print("RMSE: \t", np.sqrt(mean_squared_error(y_actual, y_predict)))
    print(" r^2 : \t", explained_variance_score(y_actual, y_predict))


# **Linear Regression**

# In[77]:


from sklearn.linear_model import LinearRegression
linear_regression_model = LinearRegression().fit(X_train,y_train)


print (linear_regression_model.intercept_)
# print(linear_regression_model.coef_ )
 
# Lets print the summary model
summary_LR_Model = pd.DataFrame(linear_regression_model.coef_,X_train.columns,columns=['coeff'])
print(summary_LR_Model)


# In[78]:


predict_LR_train = linear_regression_model.predict(X_train)
RMSE(y_train,predict_LR_train)

print()
predict_LR = linear_regression_model.predict(X_test)
RMSE(y_test,predict_LR,False)


# **Decision tree**

# In[79]:


from sklearn.tree import DecisionTreeRegressor

# DT = DecisionTreeRegressor(max_depth=10,min_samples_split=9,max_leaf_nodes=49)
DT = DecisionTreeRegressor()
DT.fit(X_train,y_train)


# In[80]:


predict_DT_train = DT.predict(X_train)
RMSE(y_train,predict_DT_train)

print()

predictions_DT = DT.predict(X_test)
RMSE(y_test,predictions_DT,False)


# In[81]:


##Overfitting and Grid search

from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2,20)),
         'min_samples_split': list(range(2,10)),
         'max_depth': list(range(2,10))}


gs_CV_DT = GridSearchCV(DecisionTreeRegressor(random_state =42),params,n_jobs=1, verbose= 1)

gs_CV_DT.fit(X_train,y_train)

print(gs_CV_DT.best_estimator_)


# In[82]:


predict_DT_train = gs_CV_DT.predict(X_train)
RMSE(y_train,predict_DT_train)

print()


predict_GS = gs_CV_DT.predict(X_test)

RMSE(y_test,predict_GS,False)


# ### *Random Forest*

# In[83]:


from sklearn.ensemble import RandomForestRegressor

RF_model = RandomForestRegressor(n_estimators=500, random_state= 43 ).fit(X_train,y_train)

print(RF_model)


# In[84]:


# Predict the model using predict funtion
RF_predict_train = RF_model.predict(X_train)
RMSE(y_train,RF_predict_train)

print()

RF_predict= RF_model.predict(X_test)
RMSE(y_test,RF_predict,False)


# In[100]:


#Printing Feature importance of the model
feat_importances = pd.Series(RF_model.feature_importances_, index=X_train.columns)
feat_importances.plot(kind='bar')


# In[90]:


##Random forest with GridSearchCV

params = {'max_depth': list(range(2,20)),
         'min_samples_split': list(range(2,10)) ,
         'max_leaf_nodes': list(range(2,10))}

gs_CV_RF = GridSearchCV(RandomForestRegressor(random_state =42),params,n_jobs=1, verbose= 1)

gs_CV_RF.fit(X_train,y_train)

#print(gs_CV_RF.best_estimator_)


# In[91]:


# Predict the model using predict funtion
predict_GS_RF = gs_CV_RF.predict(X_train)
RMSE(y_train,predict_GS_RF)

print()


predict_GS_RF = gs_CV_RF.predict(X_test)
RMSE(y_test,predict_GS_RF,False)


# ### *Gradient Boost Regressor*

# In[92]:


from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor().fit(X_train,y_train)
print(GBR)

predict_GBR_train = GBR.predict(X_train)
RMSE(y_train, predict_GBR_train)

print()
predict_GBR = GBR.predict(X_test)
RMSE(y_test,predict_GBR,False)


# In[107]:


##Graiendt Boost Regressor with GridSearchCV 

params = {
         'max_depth': list(range(2,20,2)),
          'min_samples_split' :list(range(5,10)),
    'max_leaf_nodes': list(range(5,10))}

gs_CV_GBR = GridSearchCV(GradientBoostingRegressor(random_state =42),params,n_jobs=1, verbose= 1, cv =5)

gs_CV_GBR.fit(X_train,y_train)


# In[108]:


predict_GBR_CV_train = gs_CV_GBR.predict(X_train)
RMSE(y_train, predict_GBR_CV_train)

print()
predict_GBR_CV = gs_CV_GBR.predict(X_test)
RMSE(y_test,predict_GBR_CV,False)


# Compare to all other model, gradient boost model has 83.72% accuracy and RMSE is 2.05 and r^2 is 0.74.
# Hence, Gradient boost is best model for cab fare prediction

# In[ ]:


test = test.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude'],axis = 1)

test.head(2)


# In[ ]:


##Cab fare prediction for test data
test['fare_amount'] = GBR.predict(test)


# In[ ]:


test.to_csv('cabFarePrediction1.csv', index=False)


# In[ ]:





# In[ ]:




