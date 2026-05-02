import os
import time
from dotenv import load_dotenv

load_dotenv()

# Try Gemini first
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

gemini_model = None
groq_client = None

# ---------------- GEMINI SETUP ----------------
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")
    except Exception:
        gemini_model = None

# ---------------- GROQ SETUP ----------------
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None


def build_prompt(resume_text, jd_text, score):
    return f"""
You are an expert recruiter.

Match Score: {score}%

JOB DESCRIPTION:
{jd_text[:1000]}

RESUME:
{resume_text[:1000]}

Give:
- Why this score
- Key strengths
- Key gaps
- Final recommendation

Keep it concise (4–6 lines).
"""


def get_feedback(resume_text: str, jd_text: str, score: float) -> str:

    prompt = build_prompt(resume_text, jd_text, score)

    # ---------------- TRY GEMINI ----------------
    # if gemini_model:
    #     for attempt in range(3):
    #         try:
    #             response = gemini_model.generate_content(prompt)
    #             return response.text
    #         except Exception as e:
    #             if "429" in str(e):
    #                 time.sleep(10)
    #             else:
    #                 break

    # ---------------- FALLBACK TO GROQ ----------------
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq error: {e}"

    # ---------------- FINAL FALLBACK ----------------
    return "Feedback unavailable: No valid API configured."