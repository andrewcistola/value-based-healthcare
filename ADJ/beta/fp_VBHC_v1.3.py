# FractureProof v1.3
## Value Based Healthcare Reimbursements

### Import Python Libraries
import os # Operating system navigation
import pandas as pd # Widely used data manipulation library with R/Excel like tables named 'data frames'
import numpy as np # Widely used matrix library for numerical processes
import statsmodels.api as sm # Statistics package best for regression models for statistical tests
from sklearn.preprocessing import StandardScaler # Standard scaling for easier use of machine learning algorithms
from sklearn.impute import SimpleImputer # Univariate imputation for missing data
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
title = "fp_VBHC_v1.3"
path = "fp/VBHC/v1.3/"

## Section A: Collect Possible Predictors from Public Access Data

### Import Data
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms["train"] = np.where(df_cms["quant"] <= 1, 1, 0) # Create categorical test target outcome based on conditions
df_cms["test"] = np.where(df_cms["2019 VBP Adjustment Factor"] <= 1, 1, 0) # Create categorical train target outcome based on conditions
df_cms = df_cms.drop(columns = ["Facility ID", "FIPS"]) # Drop ID variables
df_cms = df_cms.drop(columns = ["2018 VBP Adjustment Factor", "2019 VBP Adjustment Factor", "2020 VBP Adjustment Factor"]) # Drop proximity features: Adjustment factor scores
df_cms = df_cms.drop(columns = ["Total Performance Score", "Weighted Normalized Clinical Outcomes Domain Score", "Weighted Safety Domain Score", "Weighted Person and Community Engagement Domain Score", "Weighted Efficiency and Cost Reduction Domain Score"]) # Drop proximity features: Adjustment factor scores

### Data Pre-processing
df_prep = df_cms
quant = df_prep.pop("quant") # Remove quantitative outcome
train = df_prep.pop("train") # Remove quantitative outcome
test = df_prep.pop("test") # Remove quantitative outcome
df_prep = df_prep.dropna(axis = 1, thresh = 0.75*len(df_prep)) # Drop features less than 75% non-NA count for all columns
df_prep = pd.DataFrame(SimpleImputer(strategy = "median").fit_transform(df_prep), columns = df_prep.columns) # Impute missing data
df_prep.insert(0, "quant", quant) # Reattach qunatitative outcome to front of data frame
df_prep.insert(0, "train", train) # Reattach outcome
df_prep.insert(0, "test", test) # Reattach outcome
df_prep = df_prep.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
df_prep = pd.DataFrame(StandardScaler().fit_transform(df_prep.values), columns = df_prep.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.

### Principal Component Analysis
df_pca = df_prep.drop(columns = ["quant", "train", "test"]) # Drop outcomes and targets
degree = len(df_pca.columns) - 2 # Save number of features -1 to get degrees of freedom
pca = PCA(n_components = degree) # Pass the number of components to make PCA model based on degrees of freedom
pca.fit(df_pca) # Fit initial PCA model
df_comp = pd.DataFrame(pca.explained_variance_) # Print explained variance of components
df_comp = df_comp[(df_comp[0] > 1)] # Save eigenvalues above 1 to identify components
components = len(df_comp.index) - 3 # Save count of components for Variable reduction
pca = PCA(n_components = components) # you will pass the number of components to make PCA model
pca.fit_transform(df_pca) # finally call fit_transform on the aggregate data to create PCA results object
df_pc = pd.DataFrame(pca.components_, columns = df_pca.columns) # Export eigenvectors to data frame with column names from original data
df_pc["Variance"] = pca.explained_variance_ratio_ # Save eigenvalues as their own column
df_pc = df_pc[df_pc["Variance"] > df_pc["Variance"].mean()] # Susbet by eigenvalues with above average exlained variance ratio
df_pc = df_pc.abs() # Get absolute value of eigenvalues
df_pc = df_pc.drop(columns = ["Variance"]) # Drop outcomes and targets
df_p = pd.DataFrame(df_pc.max(), columns = ["MaxEV"]) # select maximum eigenvector for each feature
df_p = df_p[df_p.MaxEV > df_p.MaxEV.mean()] # Susbet by above average max eigenvalues 
df_p = df_p.reset_index() # Add a new index of ascending values, existing index consisting of feature labels becomes column named "index"
df_pca = df_p.rename(columns = {"index": "Features"}) # Rename former index as features

### Random Forest Regressor
X = df_prep.drop(columns = ["quant", "train", "test"]) # Drop outcomes and targets
Y = df_prep["quant"] # Isolate Outcome variable
forest = RandomForestRegressor(n_estimators = 1000, max_depth = 10) #Use default values except for number of trees. For a further explanation see readme included in repository. 
forest.fit(X, Y) # Fit Forest model, This will take time
rf = forest.feature_importances_ # Output importances of features
l_rf = list(zip(X, rf)) # Create list of variables alongside importance scores 
df_rf = pd.DataFrame(l_rf, columns = ["Features", "Gini"]) # Create data frame of importances with variables and gini column names
df_rf = df_rf[(df_rf["Gini"] > df_rf["Gini"].mean())] # Subset by Gini values higher than mean

### Recursive Feature Elimination
df_pca_rf = pd.merge(df_pca, df_rf, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
pca_rf = df_pca_rf["Features"].tolist() # Save features from data frame
X = df_prep[pca_rf] # Save features columns as predictor data frame
Y = df_prep["quant"] # Selected quantitative outcome from original data frame
recursive = RFECV(estimator = LinearRegression(), min_features_to_select = 5) # define selection parameters, in this case all features are selected. See Readme for more ifo
recursive.fit(X, Y) # This will take time
rfe = recursive.support_ # Save Boolean values as numpy array
l_rfe = list(zip(X, rfe)) # Create list of variables alongside RFE value 
df_rfe = pd.DataFrame(l_rfe, columns = ["Features", "RFE"]) # Create data frame of importances with variables and gini column names
df_rfe = df_rfe[df_rfe.RFE == True] # Select Variables that were True

### Multiple Regression
pca_rf_rfe = df_rfe["Features"].tolist() # Save chosen featres as list
X = df_prep.filter(pca_rf_rfe) # Keep only selected columns from rfe
Y = df_prep["quant"] # Add outcome variable
regression = LinearRegression() # Linear Regression in scikit learn
regression.fit(X, Y) # Fit model
coef = regression.coef_ # Coefficient models as scipy array
l_reg = list(zip(X, coef)) # Create list of variables alongside coefficient 
df_reg = pd.DataFrame(l_reg, columns = ["Features", "Coefficients"]) # Create data frame of importances with variables and gini column names

### Export Results
df_final = pd.merge(df_pca_rf, df_reg, on = "Features", how = "inner") # Join by column while keeping only items that exist in both, select outer or left for other options
final = df_final["Features"].tolist() # Save chosen featres as list
print(df_final) # Show in terminal
df_final.to_csv(path + title + ".csv") # Export df as csv

## Section 2: Modeling and Prediction

### Reimport Original Data
df_cms = pd.read_csv("hnb/CMS/CMS_2018_FIPS_full.csv", low_memory = 'false') # Import dataset saved as csv in _data folder
df_cms = df_cms.rename(columns = {"2020 VBP Adjustment Factor": "quant"}) # Rename quantitative outcome
df_cms["train"] = np.where(df_cms["2019 VBP Adjustment Factor"] <= 1, 1, 0) # Create categorical train target outcome based on conditions
df_cms["test"] = np.where((df_cms["quant"] <= 1), 1, 0) # Create categorical test target outcome based on conditions
df_cms["test2"] = np.where(df_cms["2018 VBP Adjustment Factor"] <= 1, 1, 0) # Create categorical train target outcome based on conditions

### Build Model of predictive features by hand
features = ["Hospital Ownership ForProfit", 
            "Hospital overall rating",
            "Patients who gave their hospital a rating of 9 or 10 on a scale from 0 (lowest) to 10 (highest)", 
            "Medicare hospital spending per patient (Medicare Spending per Beneficiary)",
            "Rate of readmission after discharge from hospital (hospital-wide)",
            "TOTAL HAC SCORE",
            "Average (median) time patients spent in the emergency department before leaving from the visit A lower number of minutes is better",
            "Serious complications",
            "Heart failure (HF) 30-Day Readmission Rate",
            "Pneumonia (PN) 30-Day Readmission Rate",
            "Payment for pneumonia patients",
            "Payment for heart failure patients", 
            "quant",
            "train",
            "test",
            "test2"] # Hand select features from results table
df_sub = df_cms.filter(features) # Subset by hand selected features for model
df_sub = df_sub.dropna() # Drop all rows with NA values (should be none, this is just to confirm)
X = df_sub.drop(columns = ["quant", "train", "test"]) # features as x
Y = df_sub["quant"] # Save outcome variable as y
mod = sm.OLS(Y, X) # Describe linear model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

### Build neural netowkr to predict outcomes
df_sub = pd.DataFrame(StandardScaler().fit_transform(df_sub.values), columns = df_sub.columns) # Standard scale values by converting the normalized features into a tabular format with the help of DataFrame.
input = df_sub.shape[1] - 4 # Save number of columns as length minus quant, test, train and round to nearest integer
nodes = round(input / 2) # Number of input dimensions divided by two for nodes in each layer
classifier = Sequential() # Sequential model building in keras
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal', input_dim = input)) # First Hidden Layer
classifier.add(Dense(nodes, activation = 'relu', kernel_initializer = 'random_normal')) # Second Hidden Layer
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal')) # Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile network with ADAM ("Adaptive moment estimation" or RMSProp + Momentum)
X = df_sub.drop(columns = ["quant", "train", "test", "test2"]) # Save features as X numpy data array

Y_train = df_sub["train"] # Save test outcome as Y numpy data array
classifier.fit(X, Y_train, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_train = (Y_train > 0)
fpr, tpr, threshold = roc_curve(Y_train, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_train = auc(fpr, tpr) # Plot ROC and get AUC score
print(auc_train)

Y_test = df_sub["test"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test = (Y_test > 0)
fpr, tpr, threshold = roc_curve(Y_test, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test = auc(fpr, tpr) # Plot ROC and get AUC score
print(auc_test)

Y_test2 = df_sub["test2"] # Save train outcome as Y numpy data array
classifier.fit(X, Y_test2, batch_size = 10, epochs = 100) # Fitting the data to the train outcome
Y_pred = classifier.predict(X) # Predict values from testing model
Y_pred = (Y_pred > 0.5)
Y_test2 = (Y_test2 > 0)
fpr, tpr, threshold = roc_curve(Y_test2, Y_pred) # Create ROC outputs, true positive rate and false positive rate
auc_test2 = auc(fpr, tpr) # Plot ROC and get AUC score
print(auc_test2)

### Append to Text File
text_file = open(path + title + ".txt", "w") # Open text file and name with subproject, content, and result suffix
text_file.write(str(res.summary())) # Line of text with space after
text_file.write("\n\n") # Add two lines of blank text at end of every section text
text_file.write("C-Statistic FY 2018 = " + str(auc_test2) + "\n") # Line of text with space after
text_file.write("C-Statistic FY 2019 = " + str(auc_train) + "\n") # Line of text with space after
text_file.write("C-Statistic FY 2020 = " + str(auc_test) + "\n") # Line of text with space after
text_file.close() # Close file





