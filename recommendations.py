
# %%
import pandas as pd


# %%
# Carregar os dados
df = pd.read_csv('medalion_data_store/silver/offer_event_features.csv')
df0 = pd.read_csv('medalion_data_store/silver/unique_event_features.csv')

#%%
popularity = df0.groupby('ofr_id_short')['cnt_offer_completed'].sum().reset_index()
#popularity = popularity.sort_values(by='cnt_offer_completed', ascending=False)
popularity

#%%
# Função para recomendar produtos populares que o usuário ainda não viu
def popularity_recommendation(person_id, df0, popularity):
    products_senn = df0[df0['person'] == person_id]['ofr_id_short'].unique()
    recommendation = popularity[~popularity['ofr_id_short'].isin(products_senn)]
    return recommendation['ofr_id_short'].head(5).tolist()

# Exemplo de recomendação para um usuário específico
usuario_exemplo = 'user_123'
print(popularity_recommendation('ffff82501cea40309d5fdd7edcca4a07', df0, popularity))















# %%
# USER-USER
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Criar matriz usuário-produto (preenchendo valores ausentes com 0)
user_item_matrix = df.pivot_table(index="person", columns="ofr_id_short", values="cnt_offer_completed", fill_value=0)

# Calcular similaridade entre usuários
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Função para recomendar produtos com base em usuários similares
def recomendar_user_user(usuario_id, user_item_matrix, user_similarity_df):
    if usuario_id not in user_item_matrix.index:
        return []  # Caso o usuário não esteja na matriz

    # Identificar usuário mais semelhante
    similar_users = user_similarity_df[usuario_id].drop(index=usuario_id).sort_values(ascending=False)
    usuario_similar = similar_users.index[0]

    # Encontrar produtos que o usuário similar consumiu e que o usuário atual não viu
    produtos_usuario = set(user_item_matrix.loc[usuario_id][user_item_matrix.loc[usuario_id] > 0].index)
    produtos_similar = set(user_item_matrix.loc[usuario_similar][user_item_matrix.loc[usuario_similar] > 0].index)
    recomendacoes = list(produtos_similar - produtos_usuario)

    return recomendacoes[:5]  # Retorna top 5 recomendações

# Exemplo de recomendação para um usuário específico
print(recomendar_user_user("user_123", user_item_matrix, user_similarity_df))

















#%%

# CONTENT-BASED
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Selecionar características do produto para recomendação (exemplo: métricas das interações)
features = ["curiosity_vr", "eagerness_cv", "overall_cr", "tran_amoun_mean"]
df_produto = df.groupby("ofr_id_short")[features].mean().reset_index()

# Normalizar as características
scaler = StandardScaler()
df_produto_scaled = df_produto.copy()
df_produto_scaled[features] = scaler.fit_transform(df_produto[features])

# Calcular similaridade entre produtos
product_similarity = cosine_similarity(df_produto_scaled[features])
product_similarity_df = pd.DataFrame(product_similarity, index=df_produto_scaled["ofr_id_short"], columns=df_produto_scaled["ofr_id_short"])

# Função para recomendar produtos similares com base nos produtos já consumidos pelo usuário
def recomendar_content_based(usuario_id, df, product_similarity_df):
    produtos_vistos = df[df["person"] == usuario_id]["ofr_id_short"].unique()
    recomendacoes = []

    for produto in produtos_vistos:
        if produto in product_similarity_df.index:
            similares = product_similarity_df[produto].sort_values(ascending=False).index[1:4].tolist()
            recomendacoes.extend(similares)

    # Retornar top 5 recomendações únicas
    return list(set(recomendacoes))[:5]

# Exemplo de recomendação para um usuário específico
print(recomendar_content_based("user_123", df, product_similarity_df))
