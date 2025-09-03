import re
import unicodedata
from typing import Tuple, Optional
import pandas as pd

# Common legal suffixes you might see in EU/US press releases.
LEGAL_SUFFIXES = [
    r"a\/s", r"ab", r"asa", r"oyj", r"oy", r"gmbh", r"ag", r"sa", r"sas", r"bv",
    r"nv", r"plc", r"ltd", r"ltda", r"inc", r"inc\.", r"corp", r"corp\.", r"co", r"co\.",
    r"llc", r"llp", r"pte", r"kk", r"k\.k\.", r"s\.p\.a\.", r"spa", r"s\.a\.b\.", r"sarl",
    r"s\.a\.r\.l\.", r"s\.r\.o\.", r"sro", r"se", r"aps", r"as", r"a\/b"  # add/trim as needed
]

def _strip_diacritics(s: str) -> str:
    """ASCII-fold (e.g., Møller -> Moller) for robust matching, but keep original for output."""
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _tokenize_company_for_regex(company_name: str) -> str:
    """
    Turn 'NKT A/S' or 'A.P. Moller - Maersk' into a regex that tolerates
    punctuation and hyphen/space variants: 'a\.p\.\s*moller\s*-\s*maersk', etc.
    """
    # ASCII-fold for matching; keep case-insensitive later.
    folded = _strip_diacritics(company_name)
    # Collapse whitespace around separators, normalize dashes.
    folded = re.sub(r"[–—−]+", "-", folded)
    folded = re.sub(r"\s+", " ", folded).strip()

    # Escape regex meta, but keep spaces/hyphens/periods flexible.
    # Replace spaces with \s+; hyphens with \s*[-–—]\s*; dots with \.?\s*
    pattern = []
    for ch in folded:
        if ch.isspace():
            pattern.append(r"\s+")
        elif ch in "-–—":
            pattern.append(r"\s*[-–—]\s*")
        elif ch == ".":
            pattern.append(r"\.?\s*")
        elif ch == "&":
            # sometimes written 'and'
            pattern.append(r"(?:&|and)")
        elif ch in "/":
            pattern.append(r"[\/\\]")
        else:
            pattern.append(re.escape(ch))
    return "".join(pattern)

def _company_variants_regex(company_name: str) -> re.Pattern:
    """
    Build a single compiled regex that matches:
      - exact company (with flexible punctuation/spacing)
      - company + legal suffix
      - legal suffix + company (rarer, but some styles)
      - common 'CompanyName - ' / 'CompanyName:' prefixes at start of title
    """
    base = _tokenize_company_for_regex(company_name)

    # Optional legal suffix after the base (with optional punctuation and spacing)
    suffix_union = "|".join(LEGAL_SUFFIXES)
    suffix = rf"(?:\s*[,\-–—:]?\s*(?:{suffix_union}))?"

    # Some press releases repeat the short form (e.g., 'NKT') separately; include a short token if clear.
    short_token = company_name.split()[0]
    short_base = _tokenize_company_for_regex(short_token) if len(short_token) >= 2 else None

    # Full company (with optional suffix)
    full_variant = rf"{base}{suffix}"

    # Short variant (optional) + optional suffix
    short_variant = rf"{short_base}{suffix}" if short_base else None

    # Whole-word boundaries (using lookarounds so we can handle punctuation)
    variants = [full_variant]
    if short_variant:
        variants.append(short_variant)

    # At title start, allow 'Company - ' or 'Company:' patterns to be eaten in one go.
    title_prefix = rf"^(?:{full_variant}|{base})\s*[\-–—:]\s*"

    combined = rf"(?i)(?<!\w)(?:{'|'.join(variants)})(?!\w)"
    return re.compile(combined), re.compile(title_prefix, flags=re.IGNORECASE)

def _normalize_spaces_quotes(s: str) -> str:
    # Normalize quotes/dashes, collapse whitespace.
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mask_company_mentions(
    text: str,
    company_name: str,
    mask: Optional[str] = "COMPANY",
    remove_title_prefix_only: bool = False,
) -> str:
    """
    Replace occurrences of the company with a mask (default 'COMPANY').
    If mask=None, occurrences are removed entirely.
    If remove_title_prefix_only=True, only strips leading 'Company - ' style prefixes (use for titles).
    """
    if not text or not company_name:
        return text or ""

    comp_re, title_prefix_re = _company_variants_regex(company_name)

    # First, drop an explicit "Company - " prefix if present (common in titles).
    text = title_prefix_re.sub("", text)

    if remove_title_prefix_only:
        # For titles where you only want to remove the prefix, return now.
        return _normalize_spaces_quotes(text)

    # Then replace all other mentions (body or title).
    replacement = (mask or "").strip()
    text = comp_re.sub(replacement, text)

    # Clean up leftover double spaces / punctuation spacing.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r"\s+([)])", r"\1", text)
    return _normalize_spaces_quotes(text)



def clean_df_from_company_names(df: pd.DataFrame, mask="COMPANY", lowercase=False) -> pd.DataFrame:
    """
    Expects columns: 'title', 'body', 'company_name'
    Adds: 'clean_title', 'clean_body'
    """
    def _clean_row(row):
        return clean_press_release(row["title_clean"], row["full_text_clean"], row["company_name"],
                                   mask=mask, lowercase=lowercase)
    
    df[["title_clean", "full_text_clean"]] = df.apply(
        lambda row: pd.Series(_clean_row(row)), axis=1
    )
    return df

def clean_press_release(
    title: str,
    body: str,
    company_name: str,
    *,
    mask: Optional[str] = "COMPANY",
    lowercase: bool = False,
) -> Tuple[str, str]:
    """
    Clean title & body by removing/masking company mentions and normalizing formatting.

    Arguments:
        title, body: original strings.
        company_name: the structured company name you store (e.g., 'NKT A/S').
        mask: 'COMPANY' (default), any string token, or None to delete mentions entirely.
        lowercase: if True, lowercases the final strings (useful for hashing/dedup keys).

    Returns:
        (clean_title, clean_body)
    """
    # Titles: remove 'Company - ' prefix first, then mask remaining mentions.
    t = mask_company_mentions(title or "", company_name, mask=mask, remove_title_prefix_only=False)
    b = mask_company_mentions(body or "", company_name, mask=mask, remove_title_prefix_only=False)

    if lowercase:
        t, b = t.lower(), b.lower()

    return t, b

# ---- Example usage ----
if __name__ == "__main__":
    company = "NKT A/S"
    title1 = 'NKT A/S - NKT has received a “Request Before the Issuing of a Decision” from the Antimonopoly Office of the Slovak Republic'
    title2 = 'NKT has received a “Request Before the Issuing of a Decision” from the Antimonopoly Office of the Slovak Republic'
    body = "Copenhagen, Denmark – NKT A/S (“NKT”) announces ... In this release, NKT confirms ..."

    ct1, cb1 = clean_press_release(title1, body, company, mask="COMPANY", lowercase=True)
    ct2, cb2 = clean_press_release(title2, body, company, mask="COMPANY", lowercase=True)

    print(ct1)
    print(ct2)
    assert ct1 == ct2  # -> titles normalize to the same string for dedup
