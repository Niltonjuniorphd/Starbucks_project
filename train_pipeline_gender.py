#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import joblib
from functions import print_metrics, feature_importance


#%%
df0 = pd.read_csv('medalion_data_store/gold/user_event_transactions.csv')

df_gender_unknow = df0[df0['gender'].isna()]


df, df_valid = train_test_split(df0, test_size=0.1, random_state=42, stratify=df0['gender'])

df_valid = df_valid.dropna()

df = df0.dropna() # drop never event completed ones


X = df.drop(columns=[
                    # 'id',
                    'person',
                    'gender',
])
y = df['gender']


#%%

print('-----Training Baseline Model------')

# defining the model pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('sel', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('scl', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop=None), X_train.select_dtypes(include=['object']).columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


# fitting the model
pipeline.fit(X_train, y_train)

# predicting
y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

# printing metrics
print_metrics(pipeline, y_train, y_test, y_pred_train, y_pred_test)
feature_selection = feature_importance(pipeline, percent=0.7,w=9, h=20)

print(feature_selection)



#%%

print('-----Tunning Model------')

# select features from the feature importance list
X = X.loc[:, feature_selection]

# set the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# defining the model parameters
param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],  # Prefix with 'classifier__'
    'classifier__max_depth': [20, 50, 100],
    'classifier__min_samples_split': [2, 3],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': ['balanced'],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0, 0.001]
}

preprocessor_grid = ColumnTransformer([
    ('sel', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('scl', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop=None), X_train.select_dtypes(include=['object']).columns)
])

pipeline_grid = Pipeline([
    ('preprocessor', preprocessor_grid),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

model_grid = pipeline_grid

grid_search = GridSearchCV(model_grid, param_grid_dt, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

# fitting the grid search
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# predicting
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# printing metrics
print_metrics(best_model, y_train, y_test, y_pred_train, y_pred_test)

# see the best parameters
pd.DataFrame(grid_search.best_params_.values(), index=[*grid_search.best_params_])


#%%
# Save the best model
joblib.dump(grid_search.best_estimator_, f'saved_models/best_model_gender_{pipeline.named_steps['classifier'].__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'saved_models/best_model_gender_{pipeline.named_steps['classifier'].__class__.__name__}.pkl')

# Check the model loaded for prediction on test data
predictions = loaded_model.predict(X_test)
predictions

# %%

# predict the valid (never seen) data
valid_predicted = loaded_model.predict(df_valid)
valid_predicted = pd.Series(valid_predicted, name='gender_predicted', index=df_valid.index)
valid_table = pd.concat([df_valid[['person', 'ofr_id_short', 'gender']], valid_predicted], axis=1)


# predicting probabilities
y_pred_test_proba = best_model.predict_proba(df_valid)
proba_df = pd.DataFrame(y_pred_test_proba, columns=best_model.classes_, index=df_valid.index)


print('---END OF MODEL---')

display(valid_table)
display(proba_df)

# %%
