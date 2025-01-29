#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Para lidar com desequilíbrios
from sklearn.preprocessing import OneHotEncoder


# Carregar os dados (assumindo o DataFrame df já carregado)
# Excluímos colunas irrelevantes e garantimos que o target seja 'gender'.

#%%
df0 = pd.read_csv('person_summary_profile.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']


X = df.drop(['person','gender'], axis=1)
y = df['gender']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#smote = SMOTE(random_state=42)
#X_balanced, y_balanced = smote.fit_resample(X.select_dtypes('number'), y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train.select_dtypes('object'))
X_test_encoded = encoder.transform(X_test.select_dtypes('object'))

scaler = StandardScaler()
X_train_scal = scaler.fit_transform(X_train_encoded)
X_test_scal = scaler.transform(X_test_encoded)

# Configuração do Random Forest com validação cruzada (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 50],
    #'min_samples_split': [2, 5],
    #'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scal, y_train)

# Melhor modelo
best_rf = grid_search.best_estimator_

# Avaliar no conjunto de teste
y_pred_test = best_rf.predict(X_test_scal)
print("Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

y_pred_train = best_rf.predict(X_train_scal)
print("Classification Report:")
print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_train))

# %%
