#HDB Resale Price Prediction Model Building and saving

import pandas as pd
import altair as alt
import plotly.express as px
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import folium
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn import pipeline
from sklearn import compose
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/hdb_data_for_prediction.csv")


# Df preprocessing

y=df['resale_price']
df=df.drop("resale_price", axis=1)


# Model Creation

numerical_feats = list(df.dtypes[df.dtypes == "float"].index)
print("Number of Numerical features: ", len(numerical_feats))

discrete_feats = list(df.dtypes[df.dtypes == "int64"].index)
print("Number of discrete features: ", len(discrete_feats))

categorical_feats = list(df.dtypes[df.dtypes == "object"].index)
print("Number of Categorical features: ", len(categorical_feats))


ct = ColumnTransformer(
    transformers=[

        ('cat', OrdinalEncoder(), categorical_feats),
        ('selector', 'passthrough', discrete_feats),
        ('num', StandardScaler(), numerical_feats)
    ],
    remainder='passthrough'
)

preprocess= Pipeline(steps=[('preprocessor', ct)])

transformed = preprocess.fit_transform(df)

columns=[]
columns.extend(categorical_feats)
columns.extend(discrete_feats)
columns.extend(numerical_feats)
X = pd.DataFrame(transformed, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##################### Linear Regression ##################################
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(X_train, y_train)
joblib.dump(LR, 'model/resalemodelLR.pkl')

# Making predictions
y_pred = LR.predict(X_test)

# Evaluating the model
lr_mse = mean_squared_error(y_test, y_pred)
lr_mae = mean_absolute_error(y_test, y_pred)
lr_rmse = np.sqrt(lr_mse) 
lr_r2 = r2_score(y_test, y_pred)
lr_intercept = LR.intercept_
lr_coef = LR.coef_
print("LR Mean Absolute Error (MAE):", lr_mae)
print("LR Root Mean Squared Error (RMSE):", lr_rmse)
print("LR R^2 Score:", lr_r2)
print("LR Intercept:", lr_intercept)
print("LR Coefficients:", lr_coef)





############################ XGBOOST ###################################

import xgboost
from sklearn.model_selection import GridSearchCV

regressor=xgboost.XGBRegressor()
param_grid = {
    'learning_rate': [0.01,0.05, 0.1, 0.2,0.3],          # Learning rate
    'max_depth': [3, 4, 5],                      # Maximum depth of the trees
    'n_estimators': [100, 200, 300,400,500]             # Number of boosting rounds
}

Best_xgbr = GridSearchCV(regressor,param_grid=param_grid,cv=5)
Best_xgbr.fit(X_train,y_train)

joblib.dump(Best_xgbr, 'model/resalemodelXGBOOST.pkl')

y_pred = Best_xgbr.predict( X_test )

xgboost_mae = mean_absolute_error(y_test, y_pred)
xgboost_mse = mean_squared_error(y_test, y_pred)
xgboost_rmse = np.sqrt(xgboost_mse)
xgboost_r2 = r2_score(y_test, y_pred)

# Print the metrics
print("XGBOOST Mean Absolute Error (MAE):", xgboost_mae)
print("XGBOOST Mean Squared Error (MSE):", xgboost_mse)
print("XGBOOST Root Mean Squared Error (RMSE):", xgboost_rmse)
print("XGBOOST R-squared (R2):", xgboost_r2)


# Number of observations and predictors
n = X_test.shape[0]
p = X_test.shape[1]

# Calculate Adjusted R-squared
xgboost_adjusted_r2 = 1 - (1-xgboost_r2)*(n-1)/(n-p-1)

print("XGBOOST Adjusted R-squared:", xgboost_adjusted_r2)





############################## Ridge ####################################3
from sklearn.linear_model import Ridge

LRidge = Ridge()

LRidge.fit(X_train, y_train)
ridge_pred = LRidge.predict(X_test)
joblib.dump(LRidge, 'model/resalemodelRidge.pkl')

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)  
ridge_r2 = r2_score(y_test, ridge_pred)

print("Ridge Mean Absolute Error (MAE):", ridge_mae)
print("Ridge Mean Squared Error (MSE):", ridge_mse)
print("Ridge Root Mean Squared Error (RMSE):", ridge_rmse)
print("Ridge R-squared (R2):", ridge_r2)











############################# Decision Tree #############################
from sklearn.tree import DecisionTreeRegressor

dtr= DecisionTreeRegressor()
dtr.fit(X_train, y_train)

joblib.dump(dtr, 'model/resalemodelDTR.pkl')

dtr_pred = dtr.predict(X_test)

dtr_mae = mean_absolute_error(y_test, dtr_pred)
dtr_mse = mean_squared_error(y_test, dtr_pred)
dtr_rmse = np.sqrt(dtr_mse) 
dtr_r2 = r2_score(y_test, dtr_pred)

print("DTR Mean Absolute Error (MAE):", dtr_mae)
print("DTR Mean Squared Error (MSE):", dtr_mse)
print("DTR Root Mean Squared Error (RMSE):", dtr_rmse)
print("DTR R-squared (R2):", dtr_r2)


############################## Random forest ####################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf_regressor = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200,300],  # Number of trees in the forest
    'max_depth': [10, 20],   # Maximum depth of the trees
}
rf_grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, verbose=2)
rf_grid_search.fit(X_train, y_train)

joblib.dump(rf_grid_search, 'model/reasalemodelRandomForest.pkl')

print(rf_grid_search.best_params_)
rf_pred = rf_grid_search.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)  
rf_r2 = r2_score(y_test, rf_pred)

print("RF Mean Absolute Error (MAE):", rf_mae)
print("RF Mean Squared Error (MSE):", rf_mse)
print("RF Root Mean Squared Error (RMSE):", rf_rmse)
print("RF R-squared (R2):", rf_r2)








#Prediction

data = [{
    'town': "YISHUN",
    'flat_type': "3 ROOM",
    'floor_area_sqm': 146.0,
    'remaining_lease':86,
    'year':2029,
    'nearest_mrt':"Ang Mo Kio",
    'mrt_dist': 0.966743,
    }]

pred_df= pd.DataFrame(data)

transformed_new_data = preprocess.transform(pred_df)

test = pd.DataFrame(transformed_new_data, columns=columns)

predicted = Best_xgbr.predict( test )
print(predicted)



# Write the result
with open('prediction_results.txt', 'w') as file:
    ##################### Linear Regression ##################################
    file.write("LR Mean Absolute Error (MAE): {}\n".format(lr_mae))
    file.write("LR Root Mean Squared Error (RMSE): {}\n".format(lr_rmse))
    file.write("LR R^2 Score: {}\n".format(lr_r2))
    file.write("LR Intercept: {}\n".format(lr_intercept))
    file.write("LR Coefficients: {}\n\n".format(lr_coef))

    ############################ XGBOOST ###################################
    file.write("XGBOOST Mean Absolute Error (MAE): {}\n".format(xgboost_mae))
    file.write("XGBOOST Mean Squared Error (MSE): {}\n".format(xgboost_mse))
    file.write("XGBOOST Root Mean Squared Error (RMSE): {}\n".format(xgboost_rmse))
    file.write("XGBOOST R-squared (R2): {}\n".format(xgboost_r2))
    file.write("XGBOOST Adjusted R-squared: {}\n\n".format(xgboost_adjusted_r2))

    ############################## Ridge ####################################
    file.write("Ridge Mean Absolute Error (MAE): {}\n".format(ridge_mae))
    file.write("Ridge Mean Squared Error (MSE): {}\n".format(ridge_mse))
    file.write("Ridge Root Mean Squared Error (RMSE): {}\n".format(ridge_rmse))
    file.write("Ridge R-squared (R2): {}\n\n".format(ridge_r2))

    ############################# Decision Tree #############################
    file.write("DTR Mean Absolute Error (MAE): {}\n".format(dtr_mae))
    file.write("DTR Mean Squared Error (MSE): {}\n".format(dtr_mse))
    file.write("DTR Root Mean Squared Error (RMSE): {}\n".format(dtr_rmse))
    file.write("DTR R-squared (R2): {}\n\n".format(dtr_r2))

    ############################## Random forest ####################################
    file.write("RF Mean Absolute Error (MAE): {}\n".format(rf_mae))
    file.write("RF Mean Squared Error (MSE): {}\n".format(rf_mse))
    file.write("RF Root Mean Squared Error (RMSE): {}\n".format(rf_rmse))
    file.write("RF R-squared (R2): {}\n\n".format(rf_r2))
