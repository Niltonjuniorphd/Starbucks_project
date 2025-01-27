#
#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from functions import train_model, tunning_model_rf, model_baseline, feature_selection, encode
import seaborn as sns
import matplotlib.pyplot as plt


#%%

df0 = pd.read_csv('transcript_b.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']

X = df[[
 'event',
 'offer_id',
 'income',
 'offer_type',
 'age_group'
  ]] 

y = df['gender']

X_enc = encode(X)

model_baseline_metrics = model_baseline(X_enc, y)
print(model_baseline_metrics)


selected_features = feature_selection(X_enc, y, res=90)

Xs = X_enc[selected_features]

model_baseline_metrics_selectedF = model_baseline(Xs, y)
print(model_baseline_metrics_selectedF)


#%%

# 'n_estimators': [50, 200],        
# 'max_depth': [None, 10, 50],       
# 'min_samples_split': [2, 5],
# 'min_samples_leaf': [1, 3]

param_grid = {
'n_estimators': [50, 100, 200],        
'max_depth': [None, 20, 50],       
'min_samples_split': [2, 4],
'min_samples_leaf': [1, 2],
'max_samples': [ None, 0.5, 0.7]
#'max_features': ['sqrt', 'log2'],
}

'''event_age = transcript_b.groupby(['person','offer_id', 'gender', 'event', 'age', 'income']).agg(
    eve_cnt = ('event', 'count'),
    avg_t = ('time', 'mean'),
    #avg_age = ('age', 'mean'),
    
    ).unstack([3]).round(1).reset_index()'''

# ['person', 'offer_id', 'gender', 'age', 'income', 'eve_cnt_offer_completed', 'eve_cnt_offer_received', 'eve_cnt_offer_viewed', 'avg_t_offer_completed', 'avg_t_offer_received', 'avg_t_offer_viewed', 'com_rate', 'view_rate', 'avg_t_diff']
# X = df[['age', 'income', 'avg_t_offer_completed', 'avg_t_offer_viewed']]
# Melhores parâmetros: {'max_depth': 20, 'max_samples': 0.7, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}

tunning_model_rf(X, y, 'f1_macro', param_grid=param_grid, cv=5)


#%%
sns.heatmap(X_enc.corr(), annot=False, cmap='coolwarm')



# %%
