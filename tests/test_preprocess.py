from src.preprocess import normalize_text

def test_normalize_text_basic():
    s = "I looooove this!!! Visit https://example.com @user #wow"
    out = normalize_text(s)
    assert "love" in out
    assert "http" not in out
    assert "user" not in out
