# tests/test_golden.py
import json
import pathlib
import pytest

DATA = pathlib.Path(__file__).resolve().parent / "data" / "golden_docs.jsonl"
EXP  = pathlib.Path(__file__).resolve().parent / "data" / "golden_expected.csv"

@pytest.mark.skipif(not (DATA.exists() and EXP.exists()), reason="golden files not present")
def test_golden_regression(plugin):
    # Very light-weight “golden” runner; compare count & simple fields
    with DATA.open() as f:
        docs = [json.loads(line) for line in f]

    got = []
    for d in docs:
        for c in plugin.detect(d):
            c["company_id"] = d.get("company_id","name:unknown")
            got.append({
                "company_id": c["company_id"],
                "metric": c["metric"],
                "unit": c["unit"],
                "period": c["period"],
                "trigger_source": c.get("_trigger_source"),
            })

    # Load expected (CSV with columns: company_id,metric,unit,period,trigger_source)
    import csv
    exp = []
    with EXP.open() as f:
        for row in csv.DictReader(f):
            exp.append(row)

    # Very soft assertion: counts match; set equality on tuples
    assert len(got) == len(exp)
    gset = {(x["company_id"], x["metric"], x["unit"], x["period"], x["trigger_source"]) for x in got}
    eset = {(x["company_id"], x["metric"], x["unit"], x["period"], x["trigger_source"]) for x in exp}
    assert gset == eset
