import re
import emoji
import nltk
from typing import Iterable, List

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return " ".join([t for t in text.split() if t not in STOPWORDS and len(t) > 2])

def normalize_batch(texts: Iterable[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
