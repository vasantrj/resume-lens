import spacy
import re
 
nlp = spacy.load("en_core_web_sm")
 
def parse_resume(text: str) -> dict:
    doc = nlp(text)
 
    # Name — first PERSON entity
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Not found")
 
    # Email
    email = next(iter(re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.I)), "Not found")
 
    # Phone
    phone = next(iter(re.findall(r"(\+?\d[\d\s\-]{8,13}\d)", text)), "Not found")
 
    # Education keywords
    edu_keywords = ["b.tech", "b.e", "m.tech", "mba", "bsc", "msc", "bachelor", "master", "phd", "diploma"]
    education = [line.strip() for line in text.splitlines()
                 if any(k in line.lower() for k in edu_keywords)]
 
    # Years of experience
    exp_match = re.findall(r"(\d+)\+?\s*years?\s*(of)?\s*experience", text, re.I)
    experience = f"{exp_match[0][0]} years" if exp_match else "Not found"
 
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "education": education[:2],
        "experience": experience
    }
 