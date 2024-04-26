#HDB Resale Price App
#Runing by -python predectionApp.py in terminal or Run directly in the IDE

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
    town = town_combobox.get()
    flat_type = flat_type_combobox.get()
    floor_area = floor_area_entry.get()
    remaining_lease = remaining_lease_entry.get()
    year = year_entry.get()
    postal_code = int(postal_code_entry.get())
    nearest_mrt = None
    mrt_dist = None
    if postal_code in full_info_df['postal'].values:
        selected_data = full_info_df[full_info_df['postal'] == postal_code]
        if selected_data['town'].iloc[0] == town:
            nearest_mrt = selected_data['nearest_mrt'].iloc[0]
            mrt_dist = selected_data['mrt_dist'].iloc[0]
        else:
            #messagebox.showinfo("Error", "The provided postcode does not correspond to the selected town. Please enter a valid postcode.")
            display_text.config(state=tk.NORMAL)
            display_text.delete(1.0, tk.END)        
            display_text.insert(tk.END, "The provided postcode does not correspond to the selected town. Please enter a valid postcode.")
    else:
        #messagebox.showinfo("Error", "Postal code not found. Please enter a valid postcode.")
        display_text.config(state=tk.NORMAL)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, "Postal code not found. Please enter a valid postcode.")



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


    model_xgboost = joblib.load('model/resalemodelXGBOOST.pkl')
    try:
        prediction = model_xgboost.predict(test)[0]
        prediction = int(prediction)
        prediction = (prediction // 100) * 100
        #messagebox.showinfo("Prediction", f"Predicted price: {prediction} SGD")
        display_text.config(state=tk.NORMAL)
        display_text.delete(1.0, tk.END)
        formatted_price = "{:,}".format(prediction)
        display_text.insert(tk.END, "Prediction: " + f"Predicted price: {formatted_price} SGD")
    except Exception as e:
        #messagebox.showerror("Error", f"An error occurred: {str(e)}")
        display_text.config(state=tk.NORMAL)
        display_text.delete(1.0, tk.END)
        display_text.insert(tk.END, f"An error occurred: {str(e)}")
 


# Main Interface
root = tk.Tk()
root.title("Input the HDB info to Get Prediction Price")

tk.Label(root, text="Town:").grid(row=0, column=0, padx=5, pady=5)
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT TIMAH', 'CHOA CHU KANG', 'CLEMENTI',
       'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
       'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN',
       'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
       'PUNGGOL',"CENTRAL AREA"]  
town_combobox = ttk.Combobox(root, values=town_options)
town_combobox.grid(row=0, column=1, padx=5, pady=5)
town_combobox.set(sample_data[0]['town'])

tk.Label(root, text="Flat Type:").grid(row=1, column=0, padx=5, pady=5)
flat_type_options = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION'] 
flat_type_combobox = ttk.Combobox(root, values=flat_type_options)
flat_type_combobox.grid(row=1, column=1, padx=5, pady=5)
flat_type_combobox.set(sample_data[0]['flat_type'])

tk.Label(root, text="Floor Area (sqm):").grid(row=2, column=0, padx=5, pady=5)
floor_area_entry = tk.Entry(root)
floor_area_entry.grid(row=2, column=1, padx=5, pady=5)
floor_area_entry.insert(0, str(sample_data[0]['floor_area_sqm']))

tk.Label(root, text="Remaining Lease:").grid(row=3, column=0, padx=5, pady=5)
remaining_lease_entry = tk.Entry(root)
remaining_lease_entry.grid(row=3, column=1, padx=5, pady=5)
remaining_lease_entry.insert(0, str(sample_data[0]['remaining_lease']))

tk.Label(root, text="Year:").grid(row=4, column=0, padx=5, pady=5)
year_entry = tk.Entry(root)
year_entry.grid(row=4, column=1, padx=5, pady=5)
year_entry.insert(0, str(sample_data[0]['year']))

tk.Label(root, text="Postal code:").grid(row=5, column=0, padx=5, pady=5)
postal_code_entry = tk.Entry(root)
postal_code_entry.grid(row=5, column=1, padx=5, pady=5)
postal_code_entry.insert(0, 560406)

submit_button = tk.Button(root, text="Press to Predict", command=predict_price)
submit_button.grid(row=6, columnspan=2, padx=5, pady=10)

display_text = tk.Text(root, height=10, width=50)
display_text.grid(row=7, columnspan=2, padx=5, pady=5)
display_text.config(state=tk.DISABLED)

root.mainloop()