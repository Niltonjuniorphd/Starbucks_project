
#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functions import ks_test


#%%
df_transc = pd.read_csv('transac_amount_sumary_profile.csv', index_col=0)
df0 = pd.read_csv('person_activity_profile.csv', index_col=0)
df0

df_unknown = df0[df0['gender'].isna()] #unknown gender to be predicted

df = df0.dropna().drop_duplicates().copy()
# df = df0.dropna(axis=0).drop_duplicates().copy() #.fillna(0)
# df = df[df['gender'] != 'O']

# %%
sns.scatterplot(
    data=df.groupby(['person', 'age_group'])[['tran_amoun_min', 'tran_amoun_max']].max(), #[df['ofr_id_short'] == 'ofr_B']
    y='tran_amoun_min',
    x='tran_amoun_max',
    hue='age_group'
    ) 
plt.xlim(0, 200)
plt.ylim(0, 200)

# %%
# Generate random examples (replace with your actual data)
feature = 'curiosity_vr'
dfa = df[df['gender'] == 'M'][feature]
dfb = df[df['gender'] == 'O'][feature]

statistic, p_value = ks_test(dfa, dfb, sig = 0.01)
sns.histplot(data=df, x=feature, hue='age_group',  bins=30, kde=True) #weights=feature,
plt.title(f'{feature} - KS test p-value: {p_value:.2f}, statistic: {statistic:.2f}')
plt.show()

# %%
sns.histplot(data=df, x=feature, hue='age_group',  bins=50) #weights=feature,

# %%
sns.boxplot(data=df, y='curiosity_vr', hue='ofr_id_short')

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

#sns.histplot(df['tran_amoun_min'], bins=50)
sns.histplot(df['tran_amoun_max'], bins=50)

# %%
red = pd.concat([df.iloc[:,4:8], df[['age_group', 'gender']]], axis=1)
#%%
sns.pairplot(red, hue='gender')
# %%

sns.barplot(data=df, y='curiosity_vr', x='gender', hue='age_group')
# %%
sns.barplot(data=df, y='reward_offer_completed', x='gender', hue='age_group')

#%%
sns.barplot(data=df, y='overall_cr', x='gender', hue='age_group')
# %%
sns.barplot(data=df, y='tran_amoun_mean', x='gender', hue='age_group')


# %%
sns.boxplot(data=df, y='tran_amoun_mean', hue='gender')
# %%
sns.boxplot(data=df_transc, y='tran_amoun_mean', x='gender', hue='gender', showfliers=False, width=0.3, legend=False)
# %%
sns.stripplot(data=df_transc, y='tran_amoun_mean', x='gender', hue='gender')

# %%
sns.catplot(data=df0, x="age_group", y="income", col="gender", aspect=.5, hue='gender', kind="box", legend=False)
# %%
g = sns.catplot(data=df0, x="age_group", y="income", col="gender", aspect=0.5, hue='gender', kind="box", legend=False)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
# %%
g = sns.catplot(data=df0, x="age_group",
                y="tran_amoun_mean", col="gender",
                aspect=0.5, hue='gender', kind="bar",
                legend=False)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
# %%
g = sns.catplot(data=df0, x="age_group",
                y="age", col="gender",
                aspect=0.5, hue='gender', kind="bar",
                legend=False)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# %%
g = sns.catplot(data=df0, x="age_group",
                y="curiosity_vr", col="gender",
                aspect=0.5, hue='gender', kind="bar",
                legend=False)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# %%
g = sns.catplot(data=df0, x="age_group",
                y="curiosity_vr", col="ofr_id_short",
                aspect=0.5, hue='gender', kind="bar",
                legend=False)

for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# %%
sns.barplot(data=df0, y='curiosity_vr', x='ofr_id_short', hue='gender')

# %%
g = sns.catplot(data=df0, x="ofr_id_short",
                y="curiosity_vr", col="age_group",
                aspect=0.5, kind="bar", hue='ofr_id_short',
                legend=False)

for ax in g.axes.flat:
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)# %%

# %%
g = sns.catplot(data=df0, x="age_group",
                y="curiosity_vr", col="ofr_id_short",
                aspect=0.5, kind="point", hue='ofr_id_short',
                legend=False)
g.fig.set_size_inches(15, 2)
for ax in g.axes.flat:
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)# %%
# %%
g = sns.catplot(data=df0, x="age_group",
                y="income", col="ofr_id_short",
                aspect=0.5, kind="point", hue='ofr_id_short',
                legend=False)
g.fig.set_size_inches(15, 2)
for ax in g.axes.flat:
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)# %%
# %%
