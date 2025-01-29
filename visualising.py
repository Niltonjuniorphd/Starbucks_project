
#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functions import ks_test


#%%
df0 = pd.read_csv('person_activity_profile.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

#df = df0.dropna().drop_duplicates().copy()
df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
df = df[df['gender'] != 'O']

# %%
sns.scatterplot(data=df[df['ofr_id_short'] == 'ofr_B'], y='tran_amoun_min', x='tran_amoun_max', hue='gender') #range(len(df))

# %%


# %%
# Generate random examples (replace with your actual data)
feature = 'eagerness_cv'
dfa = df[df['gender'] == 'F'][feature]
dfb = df[df['gender'] == 'M'][feature]

statistic, p_value = ks_test(dfa, dfb, sig = 0.01)
sns.histplot(data=df, x=feature, hue='gender',  bins=30) #weights=feature,
plt.title(f'{feature} - KS test p-value: {p_value:.2f}, statistic: {statistic:.2f}')
plt.show()

# %%
sns.histplot(data=df, x=feature, hue='ofr_id_short',  bins=50) #weights=feature,

# %%
sns.boxplot(data=df, y='age', hue='ofr_id_short')

# %%
numerical_df = df0.select_dtypes(include='number')

# Compute the correlation matrix
corr_matrix = numerical_df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=False, cmap='inferno', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Numerical Features')
plt.show()
# %%
