SKILLS_DB = [
    "python", "java", "c++", "sql", "mysql",
    "excel", "power bi", "tableau",
    "machine learning", "deep learning",
    "html", "css", "javascript",
    "react", "nodejs",
    "django", "flask",
    "aws", "azure", "git", "github",
    "docker", "kubernetes",
    "pandas", "numpy", "tensorflow"
]

def extract_skills(text):
    text = text.lower()

    found_skills = []

    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))