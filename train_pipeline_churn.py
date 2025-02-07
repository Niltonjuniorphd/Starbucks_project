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
dfa = pd.read_csv('medalion_data_store/silver/transactions_time.csv')
dfb = pd.read_csv('medalion_data_store/bronze/profile.csv')

df0 = dfa.merge(dfb, left_on=['person'], right_on=['id'], how='left')

df_gender_unknow = df0[df0['gender'].isna()]

df0 = df0.dropna()

df, df_valid = train_test_split(df0, test_size=0.1, random_state=42, stratify=df0['churn'])

X = df.drop(columns=[
                    'churn',
                    'churn2',
                    'person',
                    'id',
                    'became_member_on',
                    'bec_memb_year_month',
                    ])
X = X.drop(columns=[x for x in df0.columns if x.startswith('period')])

y = df['churn'].map({1:'c', 0:'nc'})


#%%

print('-----Training Model------')

# defining the model pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('select', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('scl', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop=None), X_train.select_dtypes(include=['object']).columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

model = pipeline

# fitting the model
model.fit(X_train, y_train)

# predicting
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# printing metrics
print_metrics(model, X_train, y_train, y_test, y_pred_train, y_pred_test)
feature_selection = feature_importance(model, w=9, h=20)
print(feature_selection)



#%%

print('-----Tunning Model------')

# select features from the feature importance list

# set the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# defining the model parameters
param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],  # Prefix with 'classifier__'
    'classifier__max_depth': [None, 50, 100],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 3],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0.0, 0.01]
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

grid_search = GridSearchCV(model_grid, param_grid_dt, cv=5, scoring='f1', n_jobs=-1, verbose=1)

# fitting the grid search
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# predicting
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# printing metrics
print_metrics(best_model, X_train, y_train, y_test, y_pred_train, y_pred_test)

# see the best parameters
pd.DataFrame(grid_search.best_params_.values(), index=[*grid_search.best_params_])


#%%
# Save the best model
joblib.dump(grid_search.best_estimator_, f'saved_models/best_model_gender_{model.named_steps['classifier'].__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'saved_models/best_model_gender_{model.named_steps['classifier'].__class__.__name__}.pkl')

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

print(valid_table)
print(proba_df)

# %%
