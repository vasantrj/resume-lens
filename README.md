# 🔍 RecruitLens

### *AI-powered Resume Screening & Intelligent Job Matching — built for speed, accuracy, and scale.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AI](https://img.shields.io/badge/AI--Powered-Groq%20%7C%20Llama3-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  What is RecruitLens?

RecruitLens is an end-to-end AI recruitment assistant that automates the most time-consuming part of hiring — reading and ranking resumes. It combines **semantic embeddings**, **NLP-based parsing**, and **LLM-driven feedback** to give recruiters a clear, data-backed view of every candidate against a job description.

No more manual shortlisting. No more missed skills. Just ranked candidates with explanations.

---

##  Why This Project Matters

Traditional resume screening is slow, biased, and inconsistent. Recruiters spend hours parsing PDFs only to miss qualified candidates buried in formatting. RecruitLens solves this by:

- **Eliminating keyword bias** — uses semantic similarity, not just keyword matching
- **Scaling effortlessly** — screen 1 or 100 resumes in seconds
- **Explaining decisions** — every ranking comes with AI-generated feedback
- **Reducing time-to-hire** — structured extraction + auto-generated candidate reports

---

##  Features

- Bulk Resume Screening | Upload multiple resumes, rank all against a job description |
- Single Resume Analysis | Deep-dive analysis of one candidate via PDF upload |
- AI Recruiter Feedback | LLM (Llama 3 via Groq) generates hiring recommendations |
- Resume Category Prediction | Auto-classifies resumes by domain (e.g., Data Science, DevOps) |
- Skill Extraction & Gap Detection | Extracts candidate skills, highlights what's missing |
- Candidate Ranking | Semantic similarity score against the job description |
- Resume Parsing | Extracts name, email, phone, and years of experience |
- PDF Report Generation | One-click downloadable candidate report |
- Interactive Visualizations | Plotly charts for match scores and skill comparisons |

---

## 🖼️ Screenshots (Added this section for now. I will add snaps soon..)

> - `screenshots/dashboard.png` → Main dashboard / bulk screening view
> - `screenshots/single_analysis.png` → Single resume analysis page
> - `screenshots/ranking_chart.png` → Candidate ranking visualization
> - `screenshots/pdf_report.png` → Sample generated PDF report

---

## 🏗️ Tech Stack

```
RecruitLens/
├── NLP & Embeddings   → Sentence Transformers (all-MiniLM-L6-v2), spaCy
├── LLM Feedback       → Groq API (Llama 3)
├── Classification     → scikit-learn (TF-IDF + ML classifier)
├── UI                 → Streamlit
├── Visualization      → Plotly
├── PDF Generation     → FPDF2
└── Language           → Python 3.10+
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vasantrj/RecruitLens.git
cd recruitlens
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set Up Environment Variables
```bash
# Create a .env file
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
recruitlens/
├── app.py                  # Streamlit entry point
├── modules/
│   ├── parser.py           # Resume parsing (spaCy NER)
│   ├── matcher.py          # Semantic similarity & ranking
│   ├── classifier.py       # Resume category prediction
│   ├── feedback.py         # Groq LLM feedback generator
│   └── report.py           # PDF report generation (FPDF2)
├── assets/
├── screenshots/
├── requirements.txt
└── .env.example
```

---

## 🔄 How It Works

```
📂 Resume(s) + Job Description
        ↓
  [spaCy NER Parser]  →  Extract name, email, phone, skills, experience
        ↓
  [Sentence Transformers]  →  Generate semantic embeddings
        ↓
  [Cosine Similarity]  →  Score & rank candidates vs. job description
        ↓
  [Groq / Llama 3]  →  Generate AI recruiter feedback per candidate
        ↓
  [Streamlit UI + Plotly]  →  Display rankings, charts, insights
        ↓
  [FPDF2]  →  Export candidate report as PDF
```

---

## 🙌 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built with 🧠 and Python · RecruitLens © 2026</p>

<p align="center">If this project helped you or you found it interesting, consider giving it a ⭐</p>
