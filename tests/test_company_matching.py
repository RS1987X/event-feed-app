import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.company_loader import load_company_names
from core.company_matcher import normalize,detect_mentioned_company_NER

def test_company_loader():
    companies = load_company_names()
    assert len(companies) > 0, "No companies loaded"
    print("✅ Loaded company count:", len(companies))
    print("🔹 First 5:", companies[:5])

def test_normalization():
    assert normalize("H&M AB") == "hm"
    assert normalize("ABB Ltd.") == "abb"
    assert normalize("Investor AB") == "investor"
    print("✅ Normalization passed")

def test_detection():
    companies = load_company_names()
    examples = [
        ("Hennes & Mauritz får höjd riktkurs", True),
        ("ABB stiger på rapport", True),
        #("Fastighetsbolag från Malmö ökar", False),
        ("Det här bolaget är en fasad", False),
        ("Absolent Air Care vinstvarnar", True),
        ("Addvise rapporterar", True),
        ("a global content and technology company", False),
        (
        "SEB spår kraftigt orderras från Saab. "
        "SEB höjer riktkursen för försvarsbolaget Saab inför rapporten "
        "för det andra kvartalet där banken räknar med stark intäktstillväxt. "
        "Samtidigt varnar analytikern för en kraftig nedgång i orderingången "
        "och aktiens höga värdering.",
        True
    ),
    ("Vitec Group vinner order", True)

        
    ]

    for text, expected in examples:
        if "Group" in text:
            print("dawdwa")
        match = detect_mentioned_company_NER(text, companies)
        assert bool(match) == expected, f"Failed on: {text}"
        print(f"✅ Detected: {match} for: '{text}'")

if __name__ == "__main__":
    print("🔍 Running tests...\n")
    #test_company_loader()
    #test_normalization()
    test_detection()
