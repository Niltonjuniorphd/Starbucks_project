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
df0 = pd.read_csv('person_activity_profile.csv', index_col=0)

df0 = df0.dropna()

df0_train, df_valid = train_test_split(df0, test_size=0.1, random_state=42, stratify=df0['ofr_id_short'])

y_valid = df_valid['ofr_id_short']

# df0 = df0[~df0['ofr_id_short'].isin(['ofr_C', 'ofr_H'])]

# df_gen_unknown = df0[(df0['gender'].isna())] #unknown gender to be predicted

# df_gender_O = df0[df0['gender'] == 'O'] # gender 'O' to be predicted

# df_no_tran = df0[df0['cnt_transaction'] == 0]  

# df = df0[(df0['gender'] != 'O') & (df0['cnt_transaction'] != 0)].dropna(axis=0)

df = df0_train.drop(columns=['person', 'became_member_on', 'bec_memb_year_month']).drop_duplicates().dropna()

# df = df[[
# 'ofr_id_short',
# 'reward_offer_completed',
# 'curiosity_vr',
# 'age',
# 'tran_amoun_tot',
# 'avg_time_transaction'
# ]]


#%%
print('-----Training Model------')

X = df.drop(['ofr_id_short'], axis=1)
# y = pd.get_dummies(df['ofr_id_short'], dtype=int).iloc[:,0]
y = df['ofr_id_short']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('select', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('num', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop='first'), X_train.select_dtypes(include=['object']).columns)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


model = pipeline

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print_metrics(model, X_train, y_train, y_test, y_pred_train, y_pred_test)
feature_selection = feature_importance(model)
print(feature_selection)

#%%

print('-----Tunning Model------')

param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'], 
    'classifier__max_depth': [None, 50, 100],
    # 'classifier__min_samples_split': [2],
    # 'classifier__min_samples_leaf': [1],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0.0, 0.01]
}
param_grid_rf = {
    'classifier__n_estimators': [100, 300],  # Number of trees in the forest
    'classifier__criterion': ['gini', 'entropy'],  # Splitting criteria
    'classifier__max_depth': [None, 100],  # Maximum depth of each tree
    'classifier__min_samples_split': [2, 5],  # Minimum samples required to split
    'classifier__min_samples_leaf': [1, 2],  # Minimum samples required in a leaf node
    'classifier__max_features': ['sqrt', 'log2'],  # Features considered at each split
    'classifier__class_weight': ['balanced', None],  # Handling class imbalance
    'classifier__bootstrap': [True, False],  # Whether to bootstrap samples
}


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
model = pipeline

grid_search = GridSearchCV(model, param_grid_dt, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

print_metrics(best_model, X_train, y_train, y_test, y_pred_train, y_pred_test)


pd.DataFrame(grid_search.best_params_.values(), index=[*grid_search.best_params_])





#%%

# Save the best model
joblib.dump(grid_search.best_estimator_, f'best_model_offer_{model.named_steps['classifier'].__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'best_model_offer_{model.named_steps['classifier'].__class__.__name__}.pkl')

# Use the model for prediction (assuming X_test is defined)
predictions = loaded_model.predict(X_test)

print(predictions)


# %%

valid_predicted = loaded_model.predict(df_valid)
valid_predicted = pd.Series(valid_predicted, name='gender_predicted', index=df_valid.index)
valid_table = pd.concat([df_valid[['person', 'ofr_id_short', 'gender']], valid_predicted], axis=1)
valid_table

# %%

y_pred_test_proba = best_model.predict_proba(df_valid)
proba_df = pd.DataFrame(y_pred_test_proba, columns=best_model.classes_, index=df_valid.index)
proba_df

# %%

def get_recommendations(proba_df):  
    recommendations = []
    for index, row in proba_df.iterrows():
        #print(row)
        top_3_indices = row.nlargest(3).index.tolist()
        recommendations.append(top_3_indices)
    return recommendations

recomendations = get_recommendations(proba_df)
recomendations

# %%
