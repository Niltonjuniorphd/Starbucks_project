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
dfa = pd.read_csv('medalion_data_store/silver/user_item_event.csv')
dfb = pd.read_csv('medalion_data_store/bronze/profile.csv')
dfc = pd.read_csv('medalion_data_store/silver/sorted_offers.csv')

df0 = dfa.merge(dfb, left_on=['person'], right_on=['id'], how='left').drop(columns=['id'])

df0 = df0.merge(dfc, on=['person'], how='left')

#%%

# df0 = pd.read_csv('medalion_data_store/gold/analytical_table.csv')

# df_gender_unknow = df0[df0['gender'].isna()]

df0 = df0.dropna()

df, df_valid = train_test_split(df0, test_size=0.1, random_state=42)

X = df.drop(columns=['person',
                    'ofr_id_short',
                    'became_member_on',
                    'bec_memb_year_month',
                    'ofr_id_short_completed',
                    'ofr_id_short_received',
                    'ofr_id_short_viewed'
                    ])

y = df['ofr_id_short']

#%%
print('-----Training Baseline Model------')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    #('sel', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    ('sca', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False), X_train.select_dtypes(include=['object']).columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


# model = pipeline

pipeline.fit(X_train, y_train)

y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)

print_metrics(pipeline, X_train, y_train, y_test, y_pred_train, y_pred_test)
feature_selection = feature_importance(pipeline, percent = 0.7,w=4, h=15)
print(feature_selection)

#%%

X = X.loc[:, feature_selection]


print('-----Tunning Model------')

param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],  # Prefix with 'classifier__'
    'classifier__max_depth': [20, 100, 300],
    'classifier__min_samples_split': [2, 3],
    'classifier__min_samples_leaf': [1, 3],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0, 0.0001]
}
param_grid_rf = {
    'classifier__n_estimators': [50, 300],  # Number of trees in the forest
    'classifier__criterion': ['entropy'],  # Splitting criteria
    'classifier__max_depth': [None, 100],  # Maximum depth of each tree
    'classifier__min_samples_split': [2],  # Minimum samples required to split
    'classifier__min_samples_leaf': [1, 2],  # Minimum samples required in a leaf node
    'classifier__max_features': ['sqrt', 'log2'],  # Features considered at each split
    'classifier__class_weight': ['balanced', None],  # Handling class imbalance
    'classifier__bootstrap': [True],  # Whether to bootstrap samples
}


preprocessor_grid = ColumnTransformer([
    ('sel', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('num', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False), X_train.select_dtypes(include=['object']).columns)
])

pipeline_grid = Pipeline([
    ('preprocessor', preprocessor_grid),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# model_grid = pipeline_grid

grid_search = GridSearchCV(pipeline_grid, param_grid_dt, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

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
joblib.dump(grid_search.best_estimator_, f'saved_models/best_model_offer_{best_model.named_steps['classifier'].__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'saved_models/best_model_offer_{best_model.named_steps['classifier'].__class__.__name__}.pkl')

# Use the model for prediction 
predictions = loaded_model.predict(X_test)

print(predictions)


# %%

# Predicting and formatting predictions
valid_predicted = loaded_model.predict(df_valid)
valid_predicted = pd.Series(valid_predicted, name='ofr_predicted', index=df_valid.index)
valid_table = pd.concat([df_valid[['person', 'ofr_id_short']], valid_predicted], axis=1)

# Getting probability predictions
y_pred_test_proba = loaded_model.predict_proba(df_valid)
proba_df = pd.DataFrame(y_pred_test_proba, columns=best_model.classes_, index=df_valid.index)

# Function to get top 3 recommendations
def get_recommendations(proba_df):  
    recommendations = proba_df.apply(lambda row: row.nlargest(5).index.tolist(), axis=1)
    return recommendations

# Generate recommendations and store in a dataframe
recommendations = get_recommendations(proba_df)
recommendations_df = pd.concat([valid_table, pd.Series(recommendations, name='recommendations', index=proba_df.index)], axis=1)

# Extract offers that were never seen before
recommendations_df['never_saw'] = recommendations_df.apply(
    lambda row: [i for i in row['recommendations'] if i != row['ofr_predicted']], axis=1
)

# Display the final dataframe
recommendations_df


display(valid_table)
display(proba_df)

# %%
