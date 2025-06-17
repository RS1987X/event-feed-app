# core/company_loader.py
import csv

def load_company_names(filepath="data/companies.csv") -> list[dict]:
    companies = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            fullname = row["Issuer Fullname"].strip()
            symbol = row["Symbol"].strip()
            if fullname:
                companies.append({
                    "name": fullname,
                    "symbol": symbol
                })
    return companies