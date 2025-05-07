import streamlit as st
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Загрузка данных и моделей
@st.cache_resource
def load_model():
    return Doc2Vec.load("recipes_doc2vec.model")

@st.cache_data
def load_data():
    df = pd.read_csv("df_sample.csv")
    df_cf = pd.read_csv("recipe_cf_embeddings.csv", index_col=0)
    df['RecipeId'] = df['RecipeId'].astype(int)
    return df, df_cf

model_doc2vec = load_model()
df, df_cf = load_data()

# Параметры
DOC2VEC_DIM = 100
CF_DIM = 50

# Рекомендательная логика

def recommend_doc2vec_only(recipe_id, topn=5):
    recipe_id = str(recipe_id)
    similar_docs = model_doc2vec.dv.most_similar(recipe_id, topn=topn+1)
    similar_ids = [int(doc_id) for doc_id, _ in similar_docs if doc_id != recipe_id][:topn]
    scores = [score for doc_id, score in similar_docs if doc_id != recipe_id][:topn]
    results = df[df['RecipeId'].isin(similar_ids)][['RecipeId', 'Name', 'Description']].copy()
    results['Score'] = scores
    return results

def recommend_hybrid(recipe_id, topn=5, alpha=0.5):
    recipe_id_str = str(recipe_id)
    recipe_id_int = int(recipe_id)

    if recipe_id_int not in df_cf.index:
        return recommend_doc2vec_only(recipe_id, topn)

    if recipe_id_str not in model_doc2vec.dv:
        return pd.DataFrame()

    vec_doc2vec = model_doc2vec.dv[recipe_id_str][:CF_DIM]
    vec_cf = df_cf.loc[recipe_id_int].values
    hybrid_vec = alpha * vec_doc2vec + (1 - alpha) * vec_cf

    common_ids = [rid for rid in df_cf.index if str(rid) in model_doc2vec.dv]
    all_doc2vec_trimmed = np.stack([model_doc2vec.dv[str(rid)][:CF_DIM] for rid in common_ids])
    all_cf_vecs = df_cf.loc[common_ids].values
    all_vecs = alpha * all_doc2vec_trimmed + (1 - alpha) * all_cf_vecs

    similarities = cosine_similarity([hybrid_vec], all_vecs)[0]
    top_indices = similarities.argsort()[::-1][1:topn+1]
    similar_ids = np.array(common_ids)[top_indices]
    similar_scores = similarities[top_indices]

    results = df[df['RecipeId'].isin(similar_ids.astype(int))][['RecipeId', 'Name', 'Description']].copy()
    results['Score'] = similar_scores
    return results

# Streamlit UI
st.title("🍽️ Рекомендатель рецептов")

recipe_options = df[['RecipeId', 'Name']].drop_duplicates().sort_values('Name')
recipe_map = {f"{row.Name} (#{row.RecipeId})": row.RecipeId for _, row in recipe_options.iterrows()}

selected_label = st.selectbox("Выберите рецепт:", list(recipe_map.keys()))
selected_id = recipe_map[selected_label]

alpha = st.slider("Учет текстов vs поведения пользователей (alpha)", 0.0, 1.0, 0.5, 0.05)
topn = st.slider("Сколько рецептов рекомендовать?", 1, 20, 5)

if st.button("🔍 Показать рекомендации"):
    with st.spinner("Готовим вкусные рекомендации..."):
        recommendations = recommend_hybrid(selected_id, topn=topn, alpha=alpha)
        if recommendations.empty:
            st.warning("Не удалось найти рекомендации. Попробуйте другой рецепт.")
        else:
            for _, row in recommendations.iterrows():
                st.subheader(f"{row['Name']} (#{row['RecipeId']})")
                st.write(row['Description'])
                st.caption(f"Score: {round(row['Score'], 3)}")
