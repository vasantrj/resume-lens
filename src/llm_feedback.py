import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
 
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.0-flash-lite")
 
def get_feedback(resume_text: str, jd_text: str, score: float) -> str:
    prompt = f"""
You are an expert recruiter. Analyze this resume against the job description.
Match Score: {score}%
 
JOB DESCRIPTION:
{jd_text[:1000]}
 
RESUME:
{resume_text[:1000]}
 
Give a 3-4 line honest recruiter feedback:
- Why this score?
- Key strengths
- Key gaps
Keep it concise and professional.
"""
    for attempt in range(3):
        try:
            response = _model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                if attempt < 2:
                    time.sleep(35)
                else:
                    return "Quota exceeded. Free tier limit reached. Try again after 24 hours."
            else:
                return f"Feedback unavailable: {e}"
 