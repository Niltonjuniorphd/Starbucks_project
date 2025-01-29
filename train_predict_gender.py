#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # Para lidar com desequilíbrios
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import joblib



# Carregar os dados (assumindo o DataFrame df já carregado)
# Excluímos colunas irrelevantes e garantimos que o target seja 'gender'.

#%%
df0 = pd.read_csv('person_activity_profile.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']

df = df[[
        'gender',
        'tran_amoun_mean',
        'tran_amoun_min',
        'tran_amoun_max',
        'tran_amoun_tot',
        'age',
        'income',
        'curiosity_vr',
        'overall_cr',
        'eagerness_cv'
 ]]

df = df.drop_duplicates()

# X = df.drop(['gender', 'became_member_on', 'bec_memb_year_month', 'tag'], axis=1)

X = df.drop(['gender'], axis=1)
y = df['gender']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

encoder = OneHotEncoder(sparse_output=False, categories='auto', drop='first')
X_train_enc = encoder.fit_transform(X_train.select_dtypes(include=['object']))
X_test_enc = encoder.transform(X_test.select_dtypes(include=['object']))

X_train_enc = pd.DataFrame(
    X_train_enc,
    columns=encoder.get_feature_names_out(),
    index=X_train.index
    )
X_test_enc = pd.DataFrame(
    X_test_enc,
    columns=encoder.get_feature_names_out(),
    index=X_test.index
    )

X_train_enc = pd.concat([X_train, X_train_enc], axis=1).drop(columns=X_train.select_dtypes(include=['object']).columns)
X_test_enc = pd.concat([X_test, X_test_enc], axis=1).drop(columns=X_test.select_dtypes(include=['object']).columns)

scaler = StandardScaler()
X_train_scal = scaler.fit_transform(X_train_enc)
X_test_scal = scaler.transform(X_test_enc)

X_train_scal = pd.DataFrame(
    X_train_scal,
    columns=scaler.get_feature_names_out(),
    index=X_train.index
    )
X_test_scal = pd.DataFrame(
    X_test_scal,
    columns=scaler.get_feature_names_out(),
    index=X_test.index
    )

print('-----Training Model------')

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scal, y_train)
y_pred_test = model.predict(X_test_scal)
y_pred_train = model.predict(X_train_scal)

print('Baseline Metrics\n')
print('Train\n')
print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_, zero_division=1))
print('Test\n')
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, zero_division=1))


print('Confusion Matrix Train\n')
cm1 = confusion_matrix(y_test, y_pred_test)
disp1 = ConfusionMatrixDisplay(
            confusion_matrix=cm1, display_labels=label_encoder.classes_)
disp1.plot(cmap="Blues")
plt.show()

print('Confusion Matrix Test\n')
cm2 = confusion_matrix(y_train, y_pred_train)
disp1 = ConfusionMatrixDisplay(
            confusion_matrix=cm2, display_labels=label_encoder.classes_)
disp1.plot(cmap="Blues")
plt.show()

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
  "Feature": X_train_scal.columns,
  "Importance": feature_importances})

importance_df = importance_df.sort_values(by="Importance", ascending=True)

importance_df["Cumulative"] = importance_df["Importance"].cumsum()

plt.figure(figsize=(4, 9))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

print(importance_df['Feature'].tolist()[::-1])

feature_selection = importance_df['Feature'].loc[(importance_df['Cumulative'] < 90)][::-1].tolist()

#%%

# Configuração do Random Forest com validação cruzada (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 50],
    #'min_samples_split': [2, 5],
    #'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

param_grid_b = {
    'n_estimators': [100],
    'max_depth': [30],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [25],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scal, y_train)

best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train_scal)
print("Classification Train Report:")
print(classification_report(y_train, y_pred_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))

y_pred_test = best_model.predict(X_test_scal)
print("Classification Test Report:")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

print("Best Parameters:", grid_search.best_params_)

#%%

# Save the best model
joblib.dump(grid_search.best_estimator_, f'best_model_{best_model.__class__.__name__}.pkl')

# Load the saved model
loaded_model = joblib.load(f'best_model_{best_model.__class__.__name__}.pkl')

# Use the model for prediction (assuming X_test is defined)
predictions = loaded_model.predict(X_test_scal)

predictions_gender = label_encoder.inverse_transform(predictions)
print(predictions_gender)
# %%
