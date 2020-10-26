# FractureProof
## Value Based Healthcare Project
### Outcome 
#### CMS Medicare Spending per Beneficiary 2018  
### Predictors
#### Selected Predictors
### Table Key
#### State County FIPS

### Set working directory to project folder
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository

### Set file title and path
title = "fp_VBHC_MSPB_BEA_FIPS_alpha"
path = "fp/VBHC/MSPB/"

### Import Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
import statsmodels.api as sm # Statistics package best for regression models for statistical tests

### CMS Data
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
features = ["Payment for heart failure patients", 
            "Payment for hip/knee replacement patients", 
            "Payment for pneumonia patients", 
            "Hospital Ownership ForProfit",
            "Medicare hospital spending per patient (Medicare Spending per Beneficiary)", 
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "Hospital overall rating",
            "FIPS"]
df_cms = df_cms.filter(features) # Drop Unwanted Columns
df_cms.info() # Get class, memory, and column info: names, data types, obs.

### ACS Data
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
features = ["DP05_0038PE_avg",
            "DP03_0097PE_avg",
            "DP02_0114PE",
            "DP02_0102PE_gini",
            "FIPS"]
df_acs = df_acs.filter(features) # Average Percent Black among zip codes in county, 
df_acs.info() # Get class, memory, and column info: names, data types, obs.

### BEA Data
df_bea = pd.read_csv("hnb/BEA/2018/BEA_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
features = ["CAINC30_230.0_2018",
            "CAINC30_280.0_2018",
            "FIPS"]
df_bea = df_bea.filter(features) # Drop Unwanted Columns
df_bea.info() # Get class, memory, and column info: names, data types, obs.

### Join Data
df_join = pd.merge(df_cms, df_acs, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_join = pd.merge(df_join, df_bea, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_join.info()

### Rename outcome and drop ID
df_out = df_join.rename(columns = {"Medicare hospital spending per patient (Medicare Spending per Beneficiary)": "outcome"}) # Rename multiple columns in place
df_out = df_out.drop(columns = ["FIPS"]) # features as x
df_out.info() # Get class, memory, and column info: names, data types, obs.

### Impute missing values
df_NA = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_out), columns = df_out.columns) # Impute missing data
df_NA = df_NA.dropna() # Drop all rows with NA values
df_NA.info() # Get class, memory, and column info: names, data types, obs.

### Linear regression: Multiple predictors
df_reg = df_NA
X = df_reg.drop(columns = ["outcome"]) # features as x
Y = df_reg["outcome"] # Save outcome variable as y
mod = sm.OLS(Y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model
df_reg.info()

## Artificial Neural Network

### Import scikit-learn: neural network
from sklearn.neural_network import MLPRegressor

### Setup ANN
X = df_reg.drop(columns = ["outcome"]) # features as x
Y = df_reg["outcome"] # Save outcome variable as y
ANN = MLPRegressor(random_state = 1, max_iter = 10000)

### Fit ANN
ANN.fit(X, Y) # Predict outcomes with off the shelf NN

### Collect ANN prediction results
predict = ANN.score(X, Y) # Get prediction score from ANN
print(predict)

## Section D: Display Results with Geographic Visuals

### Import Mapping Libraries
import folium # Mapping library with dynamic visuals
import json # library for importing and manipuation json files

### Import Shapefiles and Basemaps
map_fl = folium.Map(location = [29.6516, -82.3248], tiles = 'OpenStreetMap', zoom_start = 11) # Florida Open Street map
map_json = json.load(open("crime/crime_neighborhoods.geojson")) # Save as object



## Interactive Mapping

### Build Chorpleth
chor = choropleth(geo_data = js, data = df, columns = ["GeoID", "ColA"], threshold_scale = [100, 200], key_on = "feature.geoid", fill_color = "Blues", fill_opacity = 0.7, legend_name = "ColA Values").add_to(map) # Folium choropleth map

### Build Markers
for lat, lon, value in zip(df_["Lat"], df_["Lon"], df_["Value"]):
     fol.Marker(location = [lat, lon], popup = value, color = "blue").add_to(map) # For loop for creation of markers

### Export IM to HTML
map.save("_fig/crime_chi_map.html") # Save map as html file