import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Read the data
X_full = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Housing/CSVs/train.csv', index_col='Id')
X_test_full = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Housing/CSVs/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Furthermore we want to drop any categorical columns with missing value
# Get all categorical columns
cat_cols = [col for col in X_full if X_full[col].dtype == 'object']

# Get all categorical columns with missing values (IMPORTANT: for both train and test data!)
cat_with_missing = [col for col in cat_cols if (X_full[col].isnull().any() or X_test_full[col].isnull().any())]

# Drop the categoricals with missing values
X_full.drop(cat_with_missing, axis=1, inplace=True)
X_test_full.drop(cat_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Reassign the categorical columns remaining (to be used for OH Encoding)
cat_cols = [col for col in X_train if X_train[col].dtype == 'object']

# Investigate cardinality by getting number of unique entries in each column with categorical data
unique_vals = list(map(lambda col: X_train[col].nunique(), cat_cols))
d = dict(zip(cat_cols, unique_vals))
sorted_d = sorted(d.items(), key=lambda x: x[1])
# print(sorted_d)
# There are three columns with more than 10 unique values, OH Encoding these would significantly increase the number of
# elements in the model

# In order to not add too many new elements to the DataFrame let's only OH Encode columns with less than 10 unique values
low_cardinality = [col for col in cat_cols if X_train[col].nunique() < 10]
high_cardinality = list(set(cat_cols) - set(low_cardinality))


# Apply OH Encoder to each column with categorical data that doesn't have missing values
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality]))
oh_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality]))
oh_test = pd.DataFrame(oh_encoder.transform(X_test_full[low_cardinality]))

# Put the indices back
oh_train.index = X_train.index
oh_valid.index = X_valid.index
oh_test.index = X_test_full.index

# Remove the categorical values (will later add the OH columns instead)
X_train_num = X_train.drop(cat_cols, axis=1)
X_valid_num = X_valid.drop(cat_cols, axis=1)
X_test_num = X_test_full.drop(cat_cols, axis=1)

# Impute the numerical columns
imp = SimpleImputer()
imputed_X_train = pd.DataFrame(imp.fit_transform(X_train_num))
imputed_X_valid = pd.DataFrame(imp.transform(X_valid_num))
imputed_X_test = pd.DataFrame(imp.transform(X_test_num))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train_num.columns
imputed_X_valid.columns = X_valid_num.columns
imputed_X_test.columns = X_test_num.columns

# Put the indices back (otherwise the concat later will produce NaNs where the indices don't match)
imputed_X_train.index = X_train_num.index
imputed_X_valid.index = X_valid_num.index
imputed_X_test.index = X_test_num.index

# Put the OH Encoded columns and the imputed columns back together in one DataFrame
X_train_final = pd.concat([imputed_X_train, oh_train], axis=1)
X_valid_final = pd.concat([imputed_X_valid, oh_valid], axis=1)
X_test_final = pd.concat([imputed_X_test, oh_test], axis=1)

print(X_train_final.isnull().any())
print(X_train_final.isna().any())

# # Define and fit model
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model.fit(X_train_final, y_train)

# Save predictions to file
# output = pd.DataFrame({'Id': X_test.index,
#                        'SalePrice': preds_test})
# output.to_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Housing/PostNARemoval.csv', index=False)