import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Remove HTML, special chars, stopwords."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()  # strip HTML
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)             # keep letters only
    text = text.lower()
    tokens = [w for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(tokens)

def preprocess_dataframe(df):
    """Clean the Resume CSV dataframe."""
    # Use Resume_str if available, else parse Resume_html
    df['clean_resume'] = df['Resume_str'].fillna(df.get('Resume_html', ''))
    df['clean_resume'] = df['clean_resume'].apply(clean_text)
    df.dropna(subset=['clean_resume', 'Category'], inplace=True)
    df = df[df['clean_resume'].str.strip() != '']
    return df[['ID', 'Category', 'clean_resume']].reset_index(drop=True)