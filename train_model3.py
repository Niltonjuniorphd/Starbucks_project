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

df0 = pd.read_csv('person_summary_profile.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']


X = df.drop(['person','gender'], axis=1)

y = df['gender']

X_enc = encode(X)

model_baseline_metrics = model_baseline(X_enc, y)
print(model_baseline_metrics)


selected_features = feature_selection(X_enc, y, res=90)

Xs = X_enc[selected_features]

model_baseline_metrics_selectedF = model_baseline(Xs, y)
print(model_baseline_metrics_selectedF)


#%%

param_grid = {
'n_estimators': [50, 100, 200],        
'max_depth': [None, 20, 50],       
'min_samples_split': [2, 4],
'min_samples_leaf': [1, 2],
'max_samples': [None, 0.5, 0.7]
#'max_features': ['sqrt', 'log2'],
}

'''event_age = transcript_b.groupby(['person','offer_id', 'gender', 'event', 'age', 'income']).agg(
    eve_cnt = ('event', 'count'),
    avg_t = ('time', 'mean'),
    #avg_age = ('age', 'mean'),
    
    ).unstack([3]).round(1).reset_index()'''

tunning_model_rf(X, y, 'f1_macro', param_grid=param_grid, cv=5)


#%%
sns.heatmap(X_enc.corr(), annot=False, cmap='coolwarm')



# %%
