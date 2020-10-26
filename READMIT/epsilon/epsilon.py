# FractureProof Chaos
label = "_epsilon"
path = "VBHC/READMIT/epsilon/"
version = "FractureProof v1.4"
title = "Finding Value: Factors Associated with Hospital Payment Adjustments from CMS for FY2020"
author = "DrewC!"

## Setup Workspace

### Import python libraries
import os # Operating system navigation
from datetime import datetime
from datetime import date

### Import data science libraries
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models

### Import scikit-learn libraries
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
from sklearn.cluster import KMeans # clusters data by trying to separate samples in n groups of equal variance
from sklearn.decomposition import PCA # Principal compnents analysis from sklearn
from sklearn.ensemble import RandomForestRegressor # Random Forest regression component
from sklearn.ensemble import RandomForestClassifier # Random Forest classification component
from sklearn.feature_selection import RFECV # Recursive Feature elimination with cross validation
from sklearn.linear_model import LinearRegression # Used for machine learning with quantitative outcome
from sklearn.linear_model import LogisticRegression # Used for machine learning with quantitative outcome
from sklearn.metrics import roc_curve # Reciever operator curve
from sklearn.metrics import auc # Area under the curve 

### Import keras libraries
from keras.models import Sequential # Uses a simple method for building layers in MLPs
from keras.models import Model # Uses a more complex method for building layers in deeper networks
from keras.layers import Dense # Used for creating dense fully connected layers
from keras.layers import Input # Used for designating input layers

### Set Directory
os.chdir("/home/drewc/allocativ/") # Set wd to project repository

### Set Timestamps
day = str(date.today())
stamp = str(datetime.now())

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write("####################" + "\n\n")
text_file.write(title + "\n") # Line of text with space after
text_file.write(version + "\n") # Line of text with space after
text_file.write(author + "\n") # Line of text with space after
text_file.write(stamp + "\n") # Line of text with space after
text_file.write("\n" + "####################" + "\n\n")
text_file.close() # Close file

# Step 1: Raw Data Processing
sub = "Step 1: Raw Data Processing and Feature Engineering"
y = "Hospital Wide Readmissions after Discharge"
i = "CMS Hospital Compare 2018 release"
g = "HRSA Area Health Resource File by County 2018 release"

### Individual Features and Targets
df_raw = pd.read_csv("hnb/CMS/CMS_2018_FIPS_code.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_raw['Facility ID'] = df_raw['Facility ID'].astype("str") # Change data type of column in data frame
df_raw = df_raw.dropna(subset = ["CMS_2018_3"])
df_raw["train"] = np.where(df_raw["CMS_2018_25"] > 16.0, 1, 0) # Create categorical test target outcome based on conditions
df_raw["test"] = np.where(df_raw["CMS_2018_25"] > 16.6, 1, 0) # Create categorical test target outcome based on conditions
df_raw["quant"] = df_raw["CMS_2018_25"] # Rename multiple columns in place
df_raw = df_raw.drop(columns = ["CMS_2018_164", "CMS_2018_25"]) # Drop quantitative variables used to create target
df_raw.info() # Get class, memory, and column info: names, data types, obs.

### Export Targets
Y_raw = df_raw.filter(["FIPS", "train", "test", "quant", "Facility ID"])
Y_raw = Y_raw.set_index(["Facility ID", "FIPS", "quant"]) # Set column as index
Y_ss = pd.DataFrame(StandardScaler().fit_transform(Y_raw.values), columns = Y_raw.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
Y_q = Y_raw.reset_index(level = ["FIPS", "quant"]) # Reset Index
Y_raw = Y_raw.reset_index(level = ["Facility ID", "FIPS", "quant"]) # Reset Index
Y_ss["Facility ID"] = Y_raw["Facility ID"]
Y_ss["FIPS"] = Y_raw["FIPS"]
Y_ss["quant"] = Y_raw["quant"]
Y_ss = Y_ss.set_index(["Facility ID"]) # Set column as index
Y_quant = Y_ss["quant"]
Y_train = Y_ss["train"]
Y_test = Y_ss["test"]
Y_ss.info() # Get class, memory, and column info: names, data types, obs.

### Individual Predictors
X_i_raw = df_raw.drop(columns = ["CMS_2018_1",
                                    "CMS_2018_2",
                                    "CMS_2018_3",
                                    "CMS_2018_5", 
                                    "CMS_2018_6",
                                    "CMS_2018_7",
                                    "CMS_2018_8",
                                    "CMS_2018_9",
                                    "CMS_2018_10",
                                    "CMS_2018_11",
                                    "CMS_2018_12",
                                    "CMS_2018_13",
                                    "CMS_2018_14",
                                    "CMS_2018_15",
                                    "CMS_2018_16",
                                    "CMS_2018_17",
                                    "CMS_2018_18",
                                    "CMS_2018_19",
                                    "CMS_2018_20",
                                    "CMS_2018_21",
                                    "CMS_2018_22",
                                    "CMS_2018_23",
                                    "CMS_2018_24",
                                    "CMS_2018_26",
                                    "CMS_2018_27",
                                    "CMS_2018_28",
                                    "CMS_2018_29",
                                    "CMS_2018_30",
                                    "FIPS",
                                    "State",
                                    "train",
                                    "test"]) # Drop proximity features: Adjustment factor scores
X_i_q = X_i_raw.set_index("Facility ID") # Set column as index
X_i_x = X_i_q.drop(columns = ["quant"])
X_i_na = X_i_x.dropna(axis = 1, thresh = 0.75*len(X_i_x)) # Drop features less than 75% non-NA count for all columns
X_i_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_i_na), columns = X_i_na.columns) # Impute missing data
X_i = pd.DataFrame(StandardScaler().fit_transform(X_i_na.values), columns = X_i_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_i_q = X_i_q.reset_index(level = ["Facility ID"]) # Reset Index
X_i["Facility ID"] = X_i_q["Facility ID"]
X_i = X_i.set_index(["Facility ID"]) # Set column as index
X_i.info() # Get class, memory, and column info: names, data types, obs.

### Ecological Global Predictors
X_g_raw = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_5Y2018_code.csv") # Import dataset saved as csv in _data folder
X_g_raw = pd.merge(Y_raw, X_g_raw, on = "FIPS", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
X_g_q = X_g_raw.set_index("Facility ID") # Set column as index
X_g_q = X_g_q.drop(columns = ["FIPS", "train", "test"]) # Drop quantitative variables used to create target
X_g_x = X_g_q.drop(columns = ["quant"])
X_g_na = X_g_x.dropna(axis = 1, thresh = 0.75*len(X_g_x)) # Drop features less than 75% non-NA count for all columns
X_g_na = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(X_g_na), columns = X_g_na.columns) # Impute missing data
X_g = pd.DataFrame(StandardScaler().fit_transform(X_g_na.values), columns = X_g_na.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
X_g_q = X_g_q.reset_index(level = ["Facility ID"]) # Reset Index
X_g["Facility ID"] = X_g_q["Facility ID"]
X_g = X_g.set_index(["Facility ID"]) # Set column as index
X_g.info() # Get class, memory, and column info: names, data types, obs.X_g.info() # Get class, memory, and column info: names, data types, obs.

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub + "\n\n") # Line of text with space after
text_file.write(y + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Quantitative = Overall Readmissions after Hospital Visit" + "\n")
text_file.write("   Binary = 0/1, No/Yes" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   quant, train, test = Overall Readmissions after Hospital Visit, Above 50%, Above National Average" + "\n")
text_file.write(str(Y_raw.describe())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(i + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_i.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = Hospital" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(g + "\n") # Add two lines of blank text at end of every section text
text_file.write("   (Rows, Columns) = " + str(X_g.shape) + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Level = County" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Year = 2018 release, kept features from 2015-2018" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Feature Engineeering = 75% nonNA, Median Imputed NA, Standard Scaled" + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 2: Intital Prediction with Closed Box Models
sub2 = "Step 2: Initial Prediction with Closed Models"
m1 = "Multi-Layer Perceptron"

## Mutli-Layer Perceptron for Individual Agencies

### Build Network with keras Sequential API
# Prep Inputs
Y_train = Y_ss["train"]
Y_test = Y_ss["test"]
input = X_i.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
network.fit(X_i, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_i = network.predict(X_i) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_i > 0.5)) # Create ROC outputs, true positive rate and false positive rate
i_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_i > 0.5)) # Create ROC outputs, true positive rate and false positive rate
i_test = auc(fpr, tpr) # Plot ROC and get AUC score

## Export Intitial Prediction Results

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub2 + "\n\n") # Line of text with space after
text_file.write(i + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Epochs = 100" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = train, test" + "\n")
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(i_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(i_test) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 3: Identify Predictors with Open Box Models
sub3 = "Step 3: Identify Predictors with Open Models"
m3 = "Principal Component Analysis"
m4 = "Random Forests"
m5 = "Recursive feature Elimination"
m6 = "Multiple Regression"

## Identify Predictors for Ecological Globals

### Principal Component Analysis
degree = len(X_g.columns) - 1  # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(X_g) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(X_g) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = X_g.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features
df_pca = df_pca.sort_values(by = ["MaxEV"], ascending = False) # Sort Columns by Value

### Random Forest Regresson
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X_g, Y_quant) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X_g, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean
df_rf = df_rf.sort_values(by = ["Gini"], ascending = False) # Sort Columns by Value

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X_g[pca_rf], Y_quant) # This will take time
rfe = recursive.ranking_ # Save Boolean values as numpy array
l_rfe = list(zip(X_g[pca_rf], rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == 1] # Select Variables that were True
df_rfe = pd.merge(df_rfe, df_pca_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X_g[pca_rf_rfe], Y_quant) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X_g[pca_rf_rfe], coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names
df_reg = df_reg.sort_values(by = ["Coefficients"], ascending = False) # Sort Columns by Value

### Export feature attributes
fp_X_g = pd.merge(df_rfe, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options

### Attach Labels
df_lg = pd.read_csv("hnb/HRSA/AHRF/AHRF_2018_2019_SAS/AHRF_5Y2018_label.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_lg = df_lg.rename(columns = {"Code": "Features"})
df_fg = pd.merge(fp_X_g, df_lg, on = "Features", how = "left") # Join by column while keeping only items that exist in both, select outer or left for other options
features = df_fg["Features"].tolist()
features.append('quant')
df_g_r = X_g_q[features]
df_g_r = df_g_r.dropna()

### Build Regression Model
X = df_g_r.drop(columns = ["quant"])
Y = df_g_r["quant"]
mod = sm.OLS(Y, X) # Describe linear model
res_g = mod.fit() # Fit model

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(sub3 + "\n\n") # Line of text with space after
text_file.write(g + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Models = " + m3 + m4 + m5 + m6 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Values = Eigenvectors, Gini Impurity, True, OLS" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Thresholds = Mean, Mean, Cross Validation, All" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Outcome = quant, 2020 VBP Adjustment Factor" + "\n\n")
text_file.write(str(df_fg)  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write(str(res_g.summary())  + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file

# Step 4: Final Prediction with Open and Closed Box Models
sub4 = "Step 4: Final Prediction with Closed Box Models"
f = "Final Model of HRSA Predictors CMS Hospital Compare Data"

## Isolate Identified Predictors from Raw Data for Final Models

### Pull final feature list from raw data
f_g = fp_X_g["Features"].tolist()
X_f_g = X_g_raw[f_g]

### Join raw predictors and raw outcome data using Facility ID index
X_f_g = X_f_g.reset_index()
X_f_i = X_i_x.reset_index()
X_f_i = X_f_i.drop(columns = ["Facility ID"]) # Drop Unwanted Columns
X_f_i = X_f_i.reset_index()
Y_f = Y_raw.reset_index()
Y_f = Y_f.drop(columns = ["Facility ID"]) # Drop Unwanted Columns
df_f = pd.merge(X_f_g, X_f_i, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_f = pd.merge(df_f, Y_f, on = "index", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
df_f = df_f.drop(columns = ["FIPS", "index"]) # Drop Unwanted Columns
df_f = df_f.dropna(axis = 1, thresh = 0.75*len(df_f)) # Drop features less than 75% non-NA count for all columns
df_f = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_f), columns = df_f.columns) # Impute missing data
df_f = pd.DataFrame(StandardScaler().fit_transform(df_f.values), columns = df_f.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
df_f.info()

## Mutli-Layer Perceptron for Indetified Final Predictors

### Build Network with keras Sequential API
# Prep Inputs
ss_f = pd.DataFrame(StandardScaler().fit_transform(df_f.values), columns = df_f.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
Y_train = ss_f.filter(["train"])
Y_test = ss_f.filter(["test"])
X = ss_f
input = X.shape[1] # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
network = Sequential()
# Dense Layers
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
network.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
# Activation Layer
network.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
# Compile
network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
# Fit
final = network.fit(X, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
# Predict
Y_f = network.predict(X) # Predict values from testing model
# AUC Score
fpr, tpr, threshold = roc_curve((Y_train > 0), (Y_f > 0.5)) # Create ROC outputs, true positive rate and false positive rate
f_train = auc(fpr, tpr) # Plot ROC and get AUC score
fpr, tpr, threshold = roc_curve((Y_test > 0), (Y_f > 0.5)) # Create ROC outputs, true positive rate and false positive rate
f_test = auc(fpr, tpr) # Plot ROC and get AUC score

### Append to Text File
text_file = open(path + day + "_results" + label + ".txt", "a") # Open text file and name with subproject, content, and result suffix
text_file.write(f + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Network = " + m1 +"\n") # Add two lines of blank text at end of every section text
text_file.write("   Layers = Dense, Dense, Activation" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Functions = ReLU, ReLU, Sigmoid" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Epochs = 200" + "\n") # Add two lines of blank text at end of every section text
text_file.write("   Targets = (train, test, test2), (FY2020, FY2019, FY2018)" + "\n")
text_file.write("   AUC Scores" + "\n") # Add two lines of blank text at end of every section text
text_file.write("       train = " + str(f_train) + "\n") # Add two lines of blank text at end of every section text
text_file.write("       test = " + str(f_test) + "\n\n") # Add two lines of blank text at end of every section text
text_file.write("####################" + "\n\n")
text_file.close() # Close file
