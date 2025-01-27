#
#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from functions import train_model, tunning_model_rf
from sklearn.preprocessing import OneHotEncoder


#%%

df0 = pd.read_csv('event_age_offer_type.csv', index_col=0)
df0


df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.fillna(0).drop_duplicates().copy()
df = df[df['gender'] != 'O']

X = df.drop(columns = ['person', 'gender', 'tag'])
y = df['gender']

# for col in X.select_dtypes('float64').columns:
    # X[col] = X[col].astype('float32')

encoder = OneHotEncoder(sparse_output=False)
encoded_cols = encoder.fit_transform(X.select_dtypes(include=['object']))
X_encoded = pd.DataFrame(
    encoded_cols,
    columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns),
    index=X.index
    )
       

X = pd.concat([X, X_encoded], axis=1).drop(columns=X.select_dtypes(include=['object']).columns)


model1 = RandomForestClassifier(
    random_state=42,
    max_depth=15,
    n_estimators=100,
    min_samples_leaf=2,
    min_samples_split=10,
    max_features=0.5,
    min_impurity_decrease=0.000001,
    n_jobs=-1,
    #class_weight='balanced'
    )

model2 = DecisionTreeClassifier(
    random_state=42,
    #criterion='entropy',            # Função de impureza para divisão ('gini' ou 'entropy')
    #max_depth=100,                # Profundidade máxima da árvore (controle de overfitting)
    #min_samples_split=2,         # Número mínimo de amostras para dividir um nó
    #min_samples_leaf=2,          # Número mínimo de amostras em uma folha
    #max_features=5,           # Número máximo de features consideradas em cada divisão
    #splitter='random'            # Estratégia para escolher a divisão ('best' ou 'random')
    #class_weight='balanced'
    )

model3 = GradientBoostingClassifier(
    random_state=42,
    # n_estimators=100,
    # learning_rate=0.05,
    # max_depth=5,
    # min_samples_split=2,
    # min_samples_leaf=1,
    # #max_features='sqrt',
    # subsample=1.0,
)


result = []
for model in [model1, model2, model3]:
    print(f'--- training baseline model: {model.__class__.__name__} ---')
    y_pred_train, y_pred_test = train_model(X, y, model)
    result.append({'model': model.__class__.__name__, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test})
    
    
pd.DataFrame(result).round(2)



#%%
feature_importances = model1.feature_importances_
# Criar um dataframe para visualizar as importâncias
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
})
# Ordenar as features pela importância
importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df)

importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Calcular a importância acumulada
importance_df["Cumulative"] = importance_df["Importance"].cumsum()

# Selecionar features com importância acumulada até 95%
selected_features = importance_df[importance_df["Cumulative"] <= 0.95]["Feature"].tolist()
print(f"Selected Features: {selected_features}")


#%%

# 'n_estimators': [50, 200],        
# 'max_depth': [None, 10, 50],       
# 'min_samples_split': [2, 5],
# 'min_samples_leaf': [1, 3]

param_grid = {
'n_estimators': [50, 100, 200],        
'max_depth': [None, 20, 50],       
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 4],
'max_samples': [ None, 0.5, 0.7]
#'max_features': ['sqrt', 'log2'],
}


tunning_model_rf(X, y, 'f1_macro', param_grid=param_grid, cv=5)
# %%


#%%

import seaborn as sns
import matplotlib.pyplot as plt


#%%
sns.heatmap(df0.drop(columns=['person', 'offer_id', 'gender'], axis=1).corr(), annot=False, cmap='coolwarm')



# %%
