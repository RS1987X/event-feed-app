import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.company_loader import load_company_names
from core.company_matcher import normalize, detect_mentioned_company,detect_mentioned_company_NER

def test_company_loader():
    companies = load_company_names()
    assert len(companies) > 0, "No companies loaded"
    print("✅ Loaded company count:", len(companies))
    print("🔹 First 5:", companies[:5])

def test_normalization():
    assert normalize("H&M AB") == "hm"
    assert normalize("ABB Ltd.") == "abb ltd"
    assert normalize("Investor AB") == "investor"
    print("✅ Normalization passed")

def test_detection():
    companies = load_company_names()
    examples = [
        ("Hennes & Mauritz får höjd riktkurs", True),
        ("ABB stiger på rapport", True),
        #("Fastighetsbolag från Malmö ökar", False),
        ("Det här bolaget är en fasad", False),
        ("Air care vinstvarnar", True),
        ("Addvise rapporterar", True)
        
    ]

    for text, expected in examples:
        match = detect_mentioned_company_NER(text, companies)
        assert (match is not None) == expected, f"Failed on: {text}"
        print(f"✅ Detected: {match} for: '{text}'")

if __name__ == "__main__":
    print("🔍 Running tests...\n")
    #test_company_loader()
    #test_normalization()
    test_detection()
