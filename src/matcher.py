import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
 
@st.cache_resource(show_spinner="Loading AI model...")
def _load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
 
def match_resumes(job_description: str, df: pd.DataFrame, top_n: int = 10):
    model = _load_model()
    df = df.copy()
 
    # Batch encode all resumes at once (much faster than one-by-one)
    corpus = df["clean_resume"].tolist()
    resume_embeddings = model.encode(corpus, convert_to_tensor=True, batch_size=64, show_progress_bar=False)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
 
    scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
    df["match_score"] = [round(float(s) * 100, 2) for s in scores]
 
    return df.nlargest(top_n, "match_score").reset_index(drop=True)
 