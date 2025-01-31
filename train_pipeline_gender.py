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

df_gender_unknow = df0[df0['gender'].isna()]

df0 = df0.dropna()

df, df_valid = train_test_split(df0, test_size=0.1, random_state=42, stratify=df0['gender'])

# y_valid = df_valid['gender']
# df_valid = df_valid.drop(columns=['gender'])

X = df.drop(columns=['person', 'gender', 'became_member_on', 'bec_memb_year_month', 'channels']).drop_duplicates().dropna()
y = df['gender']


X = X[[
'tran_amoun_mean',
'income',
'age'
    ]]

#%%

print('-----Training Model------')




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('select', 'passthrough', X_train.select_dtypes(include=['number']).columns),
    #('num', StandardScaler(), X_train.select_dtypes(include=['number']).columns),
    ('cat', OneHotEncoder(sparse_output=False, drop=None), X_train.select_dtypes(include=['object']).columns)
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

# y_pred_valid = model.predict(df_valid)
# print_metrics(model, df_valid, y_valid, y_test, y_pred_valid, y_pred_test)


#%%

print('-----Tunning Model------')

param_grid_dt = {
    'classifier__criterion': ['gini', 'entropy'],  # Prefix with 'classifier__'
    'classifier__max_depth': [None, 50, 100],
    # 'classifier__min_samples_split': [2],
    # 'classifier__min_samples_leaf': [1],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_impurity_decrease': [0.0, 0.01]
}

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
joblib.dump(grid_search.best_estimator_, f'best_model_gender_{model.named_steps['classifier'].__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'best_model_gender_{model.named_steps['classifier'].__class__.__name__}.pkl')

# Use the model for prediction (assuming X_test is defined)
predictions = loaded_model.predict(X_test)
predictions

# %%

valid_predicted = loaded_model.predict(df_valid)
valid_predicted = pd.Series(valid_predicted, name='gender_predicted', index=df_valid.index)
valid_table = pd.concat([df_valid[['person', 'ofr_id_short', 'gender']], valid_predicted], axis=1)
valid_table

y_pred_test_proba = best_model.predict_proba(df_valid)
proba_df = pd.DataFrame(y_pred_test_proba, columns=best_model.classes_, index=df_valid.index)
proba_df

# %%

# def get_recommendations(proba_df):  
#     recommendations = []
#     for index, row in proba_df.iterrows():
#         #print(row)
#         top_3_indices = row.nlargest(3).index.tolist()
#         recommendations.append(top_3_indices)
#     return recommendations

# recomendations = get_recommendations(proba_df)
# recomendations
