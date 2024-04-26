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
import matplotlib.pyplot as plt


predicted_prices_1 = []
predicted_prices_2 = []

sample_data_1 = [{
    'town': "BUKIT MERAH",
    'flat_type': "5 ROOM",
    'floor_area_sqm': 115.0,
    'remaining_lease':77,
    'year':2029,
    'nearest_mrt':"Redhill",
    'mrt_dist': 0.29912,
    }]

sample_data_2 = [{
    'town': "JURONG EAST",
    'flat_type': "3 ROOM",
    'floor_area_sqm': 77.0,
    'remaining_lease':53,
    'year':2029,
    'nearest_mrt':"Jurong East",
    'mrt_dist': 1.483680,
}]

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

for year in range(2025,2035):
    sample_data_1[0]['year'] = year
    sample_data_2[0]['year'] = year
    sample_data_1[0]['remaining_lease'] = sample_data_1[0]['remaining_lease'] - 1
    sample_data_2[0]['remaining_lease'] = sample_data_2[0]['remaining_lease'] - 1
    pred_df_1= pd.DataFrame(sample_data_1)    
    transformed_new_data_1 = preprocess.transform(pred_df_1)
    test_1 = pd.DataFrame(transformed_new_data_1, columns=columns)
    pred_df_2= pd.DataFrame(sample_data_2)    
    transformed_new_data_2 = preprocess.transform(pred_df_2)
    test_2 = pd.DataFrame(transformed_new_data_2, columns=columns)

    model = joblib.load('model/resalemodelRidge.pkl')
    predicted_price_1 = model.predict(test_1)
    predicted_price_2 = model.predict(test_2)

    
    predicted_prices_1.append(predicted_price_1[0])
    predicted_prices_2.append(predicted_price_2[0])


years = list(range(2025, 2035))


plt.figure(figsize=(10, 6))
plt.plot(years, predicted_prices_1, label='High-priced HDB (BUKIT MERAH)', marker='o', color='blue')
plt.xlabel('Year')
plt.ylabel('Predicted Price SGD')
plt.title('Predicted Price Trends of High-priced HDB (BUKIT MERAH) (2025-2034)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Plot the price trend for the low-priced HDB sample (JURONG EAST)
plt.figure(figsize=(10, 6))
plt.plot(years, predicted_prices_2, label='Low-priced HDB (JURONG EAST)', marker='o', color='green')
plt.xlabel('Year')
plt.ylabel('Predicted Price SGD')
plt.title('Predicted Price Trends of Low-priced HDB (JURONG EAST) (2025-2034)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

print(predicted_prices_1)
print(predicted_prices_2)