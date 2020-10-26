# FractureProof v1.3
## Value Based Healthcare Reimbursements

### Import Python Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.cluster import KMeans # clusters data by trying to separate samples in n groups of equal variance
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 
from keras import Sequential # Sequential neural network modeling
from keras.layers import Dense # Used for creating layers within neural network

### Setup Directory and Title
os.chdir("C:/Users/drewc/GitHub/allocativ") # Set wd to project repository
title = "ACS_ZCTA_DOH_v1.3_un"
path = "fp/MSPB/alpha/"

## Section A: Collect Possible Predictors from Public Access Data

### Import Data and Clean
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_ZCTA_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_acs = df_acs[df_acs.ST == "FL"] # Susbet numeric column by condition
df_acs = df_acs.drop(columns = ["FIPS", "ST"]) # Drop Unwanted Columns
df_acs.info() # Get class, memory, and column info: names, data types, obs.

### Process Data for KMeans
df_NA = df_acs.dropna(subset = ["ZCTA"])
df_NA = df_NA.reset_index() # Reset Index
df_NA = df_NA.drop(columns = ["index"]) # Drop Unwanted Columns
df_ZCTA = df_NA.filter(["ZCTA"])
df_NA = df_NA.drop(columns = ["ZCTA"]) # Drop Unwanted Columns
df_NA = df_NA.dropna(axis = 1, thresh = 0.75*len(df_NA)) # Drop features less than 75% non-NA count for all columns
df_NA = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_NA), columns = df_NA.columns) # Impute missing data
df_NA = pd.DataFrame(StandardScaler().fit_transform(df_NA.values), columns = df_NA.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_NA["ZCTA"] = df_ZCTA["ZCTA"]
df_NA = df_NA.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_NA.info() # Get class, memory, and column info: names, data types, obs.

### PCA to determine cluster count
df_comp = df_comp.drop(columns = ["ZCTA"]) # Drop Unwanted Columns 
degree = len(df_comp.columns) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_comp) # Fit initial PCA model
df_pca = pd.DataFrame(pca.explained_variance_, columns = ["Eigenvalues"]) # Print explained variance of components
df_pca = df_pca.sort_values(by = ["Eigenvalues"], ascending = False) # Sort Columns by Value
print(df_pca) # Print value, select "elbow" to determine number of components

# K-Means Unsupervised Clustering
kmeans = KMeans(n_clusters = 5, random_state = 0) # Setup Kmeans model, pre-select number of clusters
kmeans.fit(df_comp) # Fit Kmeans
km = kmeans.labels_ # Output importances of features
l_km = list(zip(ID, km)) # Create list of variables alongside importance scores 
df_km = pd.DataFrame(l_km, columns = ["ZCTA", "Cluster"]) # Create data frame of importances with variables and gini column names
df_km["Cluster"] = df_km["Cluster"] + 1 # Add 1 to cluster array since numpy array starts at zero
df_km["Cluster"].describe() # Run descriptive statistics on all columns

### Import and Clean DOH Datat
df_doh = pd.read_csv("hnb/DOH/FL/113_5Y2018/FL_113_ZCTA.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_doh = df_doh.filter(["E00_E99_R1000", "ZCTA"]) # Filter DM2 and ZCTA
df_doh.info() # Get class, memory, and column info: names, data types, obs.

### Subset ACS by cluster
df_cl = df_km[df_km["Cluster"] == 1] # Susbet numeric column by condition
df_cl = df_km[df_km["Cluster"] == 2] # Susbet numeric column by condition
df_cl = df_km[df_km["Cluster"] == 3] # Susbet numeric column by condition
df_cl = df_km[df_km["Cluster"] == 4] # Susbet numeric column by condition
df_cl = df_km[df_km["Cluster"] == 5] # Susbet numeric column by condition

### Join DOH
df_cl = pd.merge(df_NA, df_cl, on = "ZCTA", how = "inner") # Merge with acs data to get ZCTAs in cluster
df_cl = pd.merge(df_doh, df_cl, on = "ZCTA", how = "inner") # Merge with DOH to get outcomes by ZCTA
df_cl = df_cl.drop(columns = ["ZCTA", "Cluster"]) # Drop ID Variables
df_cl =  df_cl.rename(columns = {"E00_E99_R1000": "quant"}) # Rename multiple columns in place
df_cl.info() # Get class, memory, and column info: names, data types, obs.

### Random Forest Regressor
Y = df_cl["quant"] # Isolate Outcome variable
X = df_cl.drop(columns = ["quant"]) # Drop outcomes and targets
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value
print(df_rf)

### Reimport Original Data Join and filter for RFE
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"Medicare hospital spending per patient (Medicare Spending per Beneficiary)": "quant"}) # Rename quantitative outcome
df_cms["test"] = np.where((df_cms["quant"] > 1), 1, 0) # Create categorical test target outcome based on conditions
df_cms = df_cms.filter(["quant", "FIPS", "test", "Hospital Ownership ForProfit", "Hospital overall rating", "Payment for heart failure patients", "Payment for hip/knee replacement patients", "Payment for pneumonia patients"])
df_acs = pd.read_csv("hnb/ACS/DP5Y2018/ACS_DP5Y2018_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_acs = df_acs.filter(["DP03_0035PE_avg", "DP02_0034PE_avg", "DP03_0136PE_avg", "DP04_0095PE_avg", "DP04_0019PE_avg", "DP03_0004PE_avg", "FIPS"])
df_doh = pd.read_csv("hnb/DOH/FL/113_5Y2018/FL_113_FIPS_gini.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_doh = df_doh.filter(["E00_E99_R1000_avg", "E00_E99_R1000_gini", "FIPS"])
df_test = pd.merge(df_doh, df_acs, on = "FIPS", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_test = pd.merge(df_cms, df_test, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
df_test = df_test.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_test = df_test.drop(columns = ["FIPS"]) # Drop ID Variables
print(df_test)

### Create Interactions
df_test["DM2_avg_2_34"] = df_test["E00_E99_R1000_avg"] * df_test["DP02_0034PE_avg"]
df_test["DM2_avg_3_04"] = df_test["E00_E99_R1000_avg"] * df_test["DP03_0004PE_avg"]
df_test["DM2_avg_4_95"] = df_test["E00_E99_R1000_avg"] * df_test["DP04_0095PE_avg"]
df_test["DM2_avg_3_136"] = df_test["E00_E99_R1000_avg"] * df_test["DP03_0136PE_avg"]
df_test["DM2_avg_4_95"] = df_test["E00_E99_R1000_avg"] * df_test["DP04_0095PE_avg"]
df_test["DM2_avg_4_19"] = df_test["E00_E99_R1000_avg"] * df_test["DP04_0019PE_avg"]
df_test["DM2_gini_2_34"] = df_test["E00_E99_R1000_gini"] * df_test["DP02_0034PE_avg"]
df_test["DM2_gini_3_04"] = df_test["E00_E99_R1000_gini"] * df_test["DP03_0004PE_avg"]
df_test["DM2_gini_4_95"] = df_test["E00_E99_R1000_gini"] * df_test["DP04_0095PE_avg"]
df_test["DM2_gini_3_136"] = df_test["E00_E99_R1000_gini"] * df_test["DP03_0136PE_avg"]
df_test["DM2_gini_4_95"] = df_test["E00_E99_R1000_gini"] * df_test["DP04_0095PE_avg"]
df_test["DM2_gini_4_19"] = df_test["E00_E99_R1000_gini"] * df_test["DP04_0019PE_avg"]
df_test.info() # Get class, memory, and column info: names, data types, obs.

### RFE
X = df_test.drop(columns = ["quant", "test"]) # Drop ID Variables
Y = df_test.filter(["quant"])
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True
print(df_rfe)

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_s1.filter(pca_rf_rfe) # Keep only selected columns from rfe
Y = df_s1["quant"] # Add outcome variable
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

### Build Model of predictive features
features = ["DM2_gini_2_34", 
            "DM2_gini_4_19", 
            "DP02_0034PE_avg",
            "DP04_0019PE_avg",
            "E00_E99_R1000_gini",
            "Hospital overall rating", 
            "quant",
            "test"] # Hand select features from results table
df_final = df_test.filter(features) # Subset by hand selected features for model
X = df_final.drop(columns = ["quant", "test"])
Y = df_final.filter(["quant"])
mod = sm.OLS(Y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Build neural netowkr to predict outcomes
df_sub = pd.DataFrame(StandardScaler().fit_transform(df_final.values), columns = df_final.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
input = df_sub.shape[1] - 2 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
X = df_sub.drop(columns = ["quant", "test"]) # Save features as X numpy data array
Y_test = df_sub.filter(["test"]) # Save train outcome as Y numpy data array
classifier.fit(X, Y_test, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
print(auc_test)







