📄 Resume Screening & Job Match Scorer
An AI-powered resume screening tool that matches resumes against a Job Description using TF-IDF and Cosine Similarity.
🚀 Live Demo

Add your Streamlit Cloud link here after deployment

📌 Problem Statement
Recruiters manually screen hundreds of resumes for each job posting. This tool automates the process by ranking resumes by relevance to a given Job Description and highlighting matched/missing keywords.
📂 Dataset

Source: Kaggle Resume Dataset
Size: 2484 resumes across 25 job categories

⚙️ How It Works
Job Description → Clean Text → TF-IDF Vector
Resumes CSV    → Clean Text → TF-IDF Matrix
                              ↓
                    Cosine Similarity Score (0–100%)
                              ↓
                    Ranked Results + Keyword Gap Analysis
🛠 Tech Stack

Python · Pandas · Scikit-learn (TF-IDF) · NLTK · Streamlit · Matplotlib · WordCloud

📁 Project Structure
Resume_Screening_Project/
├── data/           # Dataset (Resume.csv)
├── src/            # Source code
│   ├── app.py      # Streamlit app
│   ├── preprocess.py
│   ├── matcher.py
│   └── visualize.py
├── notebooks/      # EDA notebooks
├── outputs/        # Saved charts
├── docs/           # Report & PPT
└── requirements.txt
▶️ Run Locally
bashgit clone https://github.com/YOUR_USERNAME/Resume_Screening_Project
cd Resume_Screening_Project
pip install -r requirements.txt
streamlit run src/app.py
🔮 Future Scope

Replace TF-IDF with Sentence-BERT for semantic matching
Add PDF upload for custom resumes
Deploy on Streamlit Cloud

👤 Author
Your Name · LinkedIn · GitHub

-----------------------------------------------------
How to runn for now...
# 1. Activate venv first (always)
venv\Scripts\activate     # Windows

# 2. Run EDA notebook
jupyter notebook      # open 01_eda.ipynb in browser

# 3. Run Streamlit app
streamlit run src/app.py  