"""Read common tabular and line-oriented source formats."""
from pathlib import Path
import json
import pandas as pd

SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl", ".txt"}

def read_source(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    suffix = source.suffix.casefold()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported input format {suffix!r}; supported: {', '.join(sorted(SUPPORTED))}")
    if suffix == ".csv":
        try:
            return pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(source, dtype=str, keep_default_na=False, encoding="cp1258")
    if suffix == ".tsv":
        return pd.read_csv(source, sep="\\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source, dtype=str).fillna("")
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("records", [payload]))
        if not isinstance(payload, list):
            raise ValueError("JSON must contain an array of records or a data/records array.")
        return pd.json_normalize(payload).fillna("")
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return pd.json_normalize(rows).fillna("")
    return pd.DataFrame({"content": source.read_text(encoding="utf-8-sig").splitlines()})
