import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import pandas as pd
import seaborn as sns
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


sample_data = [{
    'town': "ANG MO KIO",
    'flat_type': "3 ROOM",
    'floor_area_sqm': 100.0,
    'remaining_lease':90,
    'year':2029,
    'nearest_mrt':"Ang Mo Kio",
    'mrt_dist': 0.966743,
    }]


def predict_price():
    full_info_df = pd.read_csv("data/hdb_data_with_mrt_info.csv")
    town = sample_data[0]['town']
    flat_type = sample_data[0]['flat_type']
    floor_area = sample_data[0]['floor_area_sqm']
    remaining_lease = sample_data[0]['remaining_lease']
    year = sample_data[0]['year']
    postal_code = 560460
    if postal_code in full_info_df['postal'].values:
        selected_data = full_info_df[full_info_df['postal'] == postal_code]
        if selected_data['town'].iloc[0] == town:
            nearest_mrt = selected_data['nearest_mrt'].iloc[0]
            mrt_dist = selected_data['mrt_dist'].iloc[0]
        else:
            print("The provided postcode does not correspond to the selected town. Please enter a valid postcode.")
    else:
        print("Postal code not found. Please enter a valid postcode.")



    df = pd.read_csv("data/hdb_data_for_prediction.csv")
    y=df['resale_price']
    df=df.drop("resale_price", axis=1)

    numerical_feats = list(df.dtypes[df.dtypes == "float"].index)
    discrete_feats = list(df.dtypes[df.dtypes == "int64"].index)
    categorical_feats = list(df.dtypes[df.dtypes == "object"].index)
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

    data = [{
        'town': town,
        'flat_type': flat_type,
        'floor_area_sqm': float(floor_area),
        'remaining_lease': int(remaining_lease),
        'year': int(year),
        'nearest_mrt': nearest_mrt,
        'mrt_dist': float(mrt_dist),
        }]

    pred_df= pd.DataFrame(data)    
    transformed_new_data = preprocess.transform(pred_df)
    test = pd.DataFrame(transformed_new_data, columns=columns)

    # Load the saved models
    model_xgboost = joblib.load('model/resalemodelXGBOOST.pkl')
    try:
        prediction = model_xgboost.predict(test)[0]
        prediction = int(prediction)
        prediction = (prediction // 100) * 100
        print("Prediction: " + f"Predicted price: {prediction} SGD")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

predict_price()

