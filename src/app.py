import streamlit as st
import pandas as pd
import joblib
import sys
import os
 
sys.path.insert(0, os.path.dirname(__file__))
 
from preprocess import preprocess_dataframe, clean_text
from matcher import match_resumes
from visualize import score_bar_chart, category_pie, generate_wordcloud_img
from pdf_reader import extract_text_from_pdf
from skills import extract_skills
from llm_feedback import get_feedback
from resume_parser import parse_resume
from report import generate_report
from matcher import _load_model
_load_model()  # loads once at startup, cached after
 
# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="RecruitLens AI", page_icon="⬡", layout="wide")
 
# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
 
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080C10 !important;
    color: #C8D6E5 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, div[data-testid="stToolbar"] { display: none; }
 
.stApp {
    background: #080C10 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,230,180,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(0,120,255,0.05) 0%, transparent 60%) !important;
}
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid rgba(0,230,180,0.12) !important;
}
[data-testid="stSidebar"] * { color: #8BA7C7 !important; }
[data-testid="stSidebar"] input {
    background: #131A22 !important;
    border: 1px solid rgba(0,230,180,0.2) !important;
    color: #C8D6E5 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,230,180,0.4) !important;
    color: #00E6B4 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.08em;
    font-size: 0.78rem !important;
    width: 100%;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(0,230,180,0.1) !important; }
 
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #4A6A8A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    border: none !important;
    padding: 12px 20px !important;
    transition: color 0.2s;
}
button[data-baseweb="tab"]:hover { color: #C8D6E5 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00E6B4 !important;
    border-bottom: 2px solid #00E6B4 !important;
    background: transparent !important;
}
[data-testid="stTabPanel"] { padding-top: 28px !important; }
 
[data-testid="stMetric"] {
    background: #0D1117;
    border: 1px solid rgba(0,230,180,0.12);
    border-left: 3px solid #00E6B4;
    padding: 18px 22px;
    border-radius: 2px;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #4A6A8A !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #F0F6FF !important;
}
 
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,230,180,0.5) !important;
    color: #00E6B4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    padding: 10px 24px !important;
    border-radius: 2px !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: rgba(0,230,180,0.08) !important;
    box-shadow: 0 0 20px rgba(0,230,180,0.15) !important;
    transform: translateY(-1px) !important;
}
 
textarea {
    background: #0D1117 !important;
    border: 1px solid rgba(0,230,180,0.15) !important;
    color: #C8D6E5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 2px !important;
}
textarea:focus {
    border-color: rgba(0,230,180,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,230,180,0.06) !important;
}
 
[data-testid="stSlider"] .rc-slider-track { background: #00E6B4 !important; }
[data-testid="stSlider"] .rc-slider-handle {
    background: #00E6B4 !important;
    border-color: #00E6B4 !important;
    box-shadow: 0 0 8px rgba(0,230,180,0.5) !important;
}
 
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,230,180,0.25) !important;
    background: #0D1117 !important;
    border-radius: 4px !important;
    padding: 12px !important;
}
[data-testid="stFileUploadDropzone"] { background: transparent !important; }
[data-testid="stDataFrame"] { border: 1px solid rgba(0,230,180,0.1) !important; border-radius: 2px !important; }
 
[data-testid="stDownloadButton"] > button {
    background: rgba(0,230,180,0.08) !important;
    border: 1px solid rgba(0,230,180,0.4) !important;
    color: #00E6B4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
}
 
[data-testid="stAlert"] {
    background: #0D1117 !important;
    border-radius: 2px !important;
    border-left: 3px solid #00E6B4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stSuccess { background: rgba(0,230,180,0.05) !important; border-left: 3px solid #00E6B4 !important; color: #00E6B4 !important; }
.stWarning { background: rgba(255,180,0,0.05) !important; border-left: 3px solid #FFB400 !important; color: #FFB400 !important; }
.stError   { background: rgba(255,70,70,0.05)  !important; border-left: 3px solid #FF4646 !important; }
 
hr { border-color: rgba(0,230,180,0.1) !important; margin: 28px 0 !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.01em; }
.stMarkdown h3 {
    color: #E8F0FA !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,230,180,0.12);
    padding-bottom: 8px;
    margin-top: 28px;
}
label, .stSelectbox label, .stTextArea label, .stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #4A6A8A !important;
}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:baseline;gap:16px;padding:36px 0 8px;
border-bottom:1px solid rgba(0,230,180,0.12);margin-bottom:32px;">
    <span style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
    color:#F0F6FF;letter-spacing:-0.03em;line-height:1;">RecruitLens</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#00E6B4;
    letter-spacing:0.2em;text-transform:uppercase;padding:3px 8px;
    border:1px solid rgba(0,230,180,0.35);border-radius:2px;">AI · v2.0</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#4A6A8A;
    margin-left:auto;letter-spacing:0.05em;">Resume Intelligence Platform</span>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model_files():
    model = joblib.load("src/model.pkl")
    vectorizer = joblib.load("src/vectorizer.pkl")
    return model, vectorizer
 
model, vectorizer = load_model_files()
 
# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df = preprocess_dataframe(df)
    return df
 
# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;
text-transform:uppercase;color:#00E6B4;padding:20px 0 12px;
border-bottom:1px solid rgba(0,230,180,0.12);margin-bottom:20px;">⬡ Data Source</div>
""", unsafe_allow_html=True)
 
csv_path = st.sidebar.text_input("Resume CSV Path", value="data/Resume.csv")
 
if st.sidebar.button("↻  Load Dataset"):
    try:
        st.session_state["df"] = load_dataset(csv_path)
        st.sidebar.success(f"✓  {len(st.session_state['df'])} resumes loaded")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
 
st.sidebar.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#2A4060;
margin-top:40px;padding-top:16px;border-top:1px solid rgba(0,230,180,0.06);line-height:1.6;">
MODEL · SENTENCE TRANSFORMERS<br>MATCH · COSINE SIMILARITY<br>
NLP · SPACY NER · GEMINI LLM
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["⬡  BULK SCREENING", "◈  SINGLE ANALYZER"])
 
# ══════════════════════════════════════════
# TAB 1 — BULK SCREENING
# ══════════════════════════════════════════
with tab1:
 
    if "df" not in st.session_state:
        st.markdown("""
        <div style="border:1px dashed rgba(0,230,180,0.15);border-radius:4px;
        padding:48px;text-align:center;margin:24px 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;color:#2A4060;margin-bottom:8px;">No Dataset Loaded</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#2A4060;letter-spacing:0.05em;">
                Enter CSV path in sidebar → click Load Dataset
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state["df"]
 
        c1, c2, c3 = st.columns(3)
        c1.metric("TOTAL RESUMES", len(df))
        c2.metric("CATEGORIES", df["Category"].nunique())
        c3.metric("FIELDS", len(df.columns))
 
        st.plotly_chart(category_pie(df), use_container_width=True)
        st.divider()
 
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#4A6A8A;margin-bottom:6px;">Job Description</p>', unsafe_allow_html=True)
        jd = st.text_area("", height=200, key="bulk_jd", label_visibility="collapsed",
                          placeholder="Paste the full job description here…")
 
        col_s, col_b = st.columns([1, 3])
        with col_s:
            top_n = st.slider("Top N matches", 5, 20, 10)
        with col_b:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            run = st.button("⬡  RUN MATCHING ANALYSIS")
 
        if run:
            if not jd.strip():
                st.warning("Paste a job description to continue.")
            else:
                with st.spinner("Analysing candidate pool…"):
                    results = match_resumes(jd, df, top_n)
 
                st.success(f"✓  Top {top_n} candidates identified")
                st.plotly_chart(score_bar_chart(results), use_container_width=True)
 
                st.markdown("### Match Results")
                st.dataframe(results[["ID", "Category", "match_score"]], use_container_width=True)
 
                st.markdown("### Word Cloud — Top Resumes")
                combined_text = " ".join(results["clean_resume"].tolist())
                st.plotly_chart(generate_wordcloud_img(combined_text), use_container_width=True)
 
                csv_out = results[["ID", "Category", "match_score"]].to_csv(index=False)
                st.download_button("⬇  Export Results as CSV", csv_out, "resume_matches.csv", "text/csv")
 
# ══════════════════════════════════════════
# TAB 2 — SINGLE RESUME ANALYZER
# ══════════════════════════════════════════
with tab2:
 
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#4A6A8A;margin-bottom:6px;">Upload Resume</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    with c2:
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;letter-spacing:0.15em;text-transform:uppercase;color:#4A6A8A;margin-bottom:6px;">Job Description (Optional)</p>', unsafe_allow_html=True)
        jd_single = st.text_area("", height=160, key="single_jd", label_visibility="collapsed",
                                 placeholder="Paste JD to get match score & AI feedback…")
 
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
 
        if not resume_text.strip():
            st.error("Could not extract text — ensure PDF is not scanned-only.")
        else:
            cleaned_resume = clean_text(resume_text)
            resume_vec = vectorizer.transform([cleaned_resume])
            prediction = model.predict(resume_vec)[0]
 
            st.divider()
 
            # ── Parsed Info
            parsed = parse_resume(resume_text)
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid rgba(0,230,180,0.15);
            border-radius:2px;padding:20px 24px;margin:12px 0;">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;
                font-family:'DM Mono',monospace;font-size:0.78rem;">
                    <div><span style="color:#4A6A8A;">NAME &nbsp;</span><span style="color:#F0F6FF;">{parsed['name']}</span></div>
                    <div><span style="color:#4A6A8A;">EMAIL &nbsp;</span><span style="color:#F0F6FF;">{parsed['email']}</span></div>
                    <div><span style="color:#4A6A8A;">PHONE &nbsp;</span><span style="color:#F0F6FF;">{parsed['phone']}</span></div>
                    <div><span style="color:#4A6A8A;">EXP &nbsp;</span><span style="color:#F0F6FF;">{parsed['experience']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
            # ── Category
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid rgba(0,230,180,0.15);
            border-left:4px solid #00E6B4;border-radius:2px;padding:20px 28px;
            margin:12px 0 24px;display:flex;align-items:center;gap:20px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                letter-spacing:0.2em;text-transform:uppercase;color:#4A6A8A;">Predicted Category</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#F0F6FF;">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)
 
            # ── Skills
            skills = extract_skills(cleaned_resume)
            st.markdown("### Extracted Skills")
            if skills:
                skill_html = " ".join([
                    f'<span style="display:inline-block;background:rgba(0,230,180,0.08);'
                    f'border:1px solid rgba(0,230,180,0.25);color:#00E6B4;'
                    f'font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                    f'padding:3px 10px;border-radius:2px;margin:3px 4px 3px 0;">{s}</span>'
                    for s in skills
                ])
                st.markdown(skill_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#2A4060;font-family:\'DM Mono\',monospace;font-size:0.8rem;">No recognisable skills detected.</span>', unsafe_allow_html=True)
 
            # ── Match Score + Missing Skills + Feedback
            score, missing, feedback = 0.0, [], ""
 
            if jd_single.strip():
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                cleaned_jd = clean_text(jd_single)
                jd_vec = vectorizer.transform([cleaned_jd])
                score = round(cos_sim(resume_vec, jd_vec)[0][0] * 100, 2)
 
                col_m, _ = st.columns([1, 2])
                with col_m:
                    st.metric("MATCH SCORE", f"{score}%")
 
                jd_skills = extract_skills(cleaned_jd)
                missing = list(set(jd_skills) - set(skills))
 
                st.markdown("### Missing Skills")
                if missing:
                    miss_html = " ".join([
                        f'<span style="display:inline-block;background:rgba(255,70,70,0.06);'
                        f'border:1px solid rgba(255,70,70,0.25);color:#FF6B6B;'
                        f'font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                        f'padding:3px 10px;border-radius:2px;margin:3px 4px 3px 0;">{s}</span>'
                        for s in missing
                    ])
                    st.markdown(miss_html, unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#00E6B4;font-family:\'DM Mono\',monospace;font-size:0.8rem;">✓  All required skills present.</span>', unsafe_allow_html=True)
 
                # ── LLM Feedback
                st.markdown("### 🤖 AI Recruiter Feedback")
                with st.spinner("Generating feedback…"):
                    feedback = get_feedback(resume_text, jd_single, score)
                st.markdown(f"""
                <div style="background:#0D1117;border-left:4px solid #00E6B4;
                padding:16px 20px;border-radius:2px;font-family:'DM Mono',monospace;
                font-size:0.82rem;line-height:1.7;color:#C8D6E5;">{feedback}</div>
                """, unsafe_allow_html=True)
 
            # ── PDF Report
            st.divider()
            pdf_bytes = generate_report(parsed, prediction, skills, score, missing, feedback)
            st.download_button("⬇  Download PDF Report", pdf_bytes, "candidate_report.pdf", "application/pdf")
 
# python -m streamlit run src/app.py
 