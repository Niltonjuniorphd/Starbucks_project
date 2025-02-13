#%%
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from functions import print_metrics, feature_importance, print_metrics2
from IPython.display import display

#%%
# Load data and handle missing values
df0 = pd.read_csv('../medalion_data_store/gold/user_event_transactions.csv')
df0 = df0.dropna()  # Drop incomplete records before splitting

# Train-validation split
df, df_valid = train_test_split(df0, test_size=0.1, random_state=42, stratify=df0['age_group'])
X = df.drop(columns=['person', 'age_group'])
y = df['age_group']

#%%
# Define baseline model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), X.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop='first'), X.select_dtypes(include=['object']).columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train baseline model
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
print_metrics(pipeline, y_train, y_test, y_pred_train, y_pred_test)

# Feature selection
feature_selection = feature_importance(pipeline, percent=0.7, w=4, h=10)
print(f'selected features: {feature_selection}')

#%%

# using selected features
X_selected = X[feature_selection]

# Redefine train-test split with selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Redefining processor
preprocessor_grid = ColumnTransformer([
    ('num', StandardScaler(), X_selected.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop='first'), X_selected.select_dtypes(include=['object']).columns)
])

# Redefining pipeline
pipeline_grid = Pipeline([
    ('preprocessor', preprocessor_grid),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


# Hyperparameter tuning using RandomizedSearchCV
param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [20, 50, 100],
    'classifier__min_samples_split': [2, 3],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': ['balanced'],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0.00001, 0.0001]
}


grid_search = RandomizedSearchCV(pipeline_grid, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1, n_iter=20)
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
print_metrics(best_model, y_train, y_test, y_pred_train, y_pred_test)
df_best_param = pd.DataFrame([grid_search.best_params_]).T
display(df_best_param)


#%%
# Save and reload model
model_path = f'../saved_models/model_age_group_{best_model.named_steps["classifier"].__class__.__name__}.pkl'
joblib.dump(best_model, model_path)
loaded_model = joblib.load(model_path)

#%%
# Predict on validation data
valid_predicted = loaded_model.predict(df_valid)

print_metrics2(loaded_model, y=df_valid['age_group'], y_pred=valid_predicted)

valid_table = df_valid[['person']].copy()
valid_table['age_group_original'] = df_valid['age_group']
valid_table['age_group_predicted'] = valid_predicted
display(valid_table)

#%%
# Predict probabilities
y_pred_test_proba = loaded_model.predict_proba(df_valid)
proba_df = pd.DataFrame(y_pred_test_proba, columns=loaded_model.classes_, index=df_valid.index)
display(proba_df)

print('---END OF MODEL---')
# %%
