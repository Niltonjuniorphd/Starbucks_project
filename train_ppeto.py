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
df0 = pd.read_csv('transcript_b.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.replace(np.inf, np.nan).dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']


X = df[[
 'event',
 'offer_id',
 'income',
 'offer_type',
 'age_group'
  ]] 


y = df['gender']

encoder = OneHotEncoder(sparse_output=False)
encoded_cols = encoder.fit_transform(X.select_dtypes(include=['object']))
X_encoded = pd.DataFrame(
    encoded_cols,
    columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns),
    index=X.index
    )

X_enc = pd.concat([X, X_encoded], axis=1).drop(columns=X.select_dtypes(include=['object']).columns)

X = X_enc.copy()

# Codificar a variável target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Balancear os dados de treinamento
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


# Garantir que as colunas de treino e teste sejam consistentes
X_train_balanced, X_test = X_train_balanced.align(X_test, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# Configuração do Random Forest com validação cruzada (GridSearchCV)
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Melhor modelo
best_rf = grid_search.best_estimator_

# Avaliar no conjunto de teste
y_pred = best_rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

y_pred_train = best_rf.predict(X_train_balanced)
print("Classification Report:")
print(classification_report(y_test, y_pred_train, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_train))

# %%
