"""End-to-end deterministic data preparation for ST-NLP."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from .cleaning import clean_text, normalize_text
from .categories import canonical_label, keyword_label
from .readers import read_source

TEXT_ALIASES = ("raw_content","content","text","description","feedback","report","noi_dung","phan_anh","message")
LABEL_ALIASES = ("label","category","category_name","category_code","incident_category","nhom_phan_anh","loai_phan_anh")

@dataclass
class ProcessingConfig:
    input_path: str
    output_dir: str = "ml/data/processed"
    text_column: str | None = None
    label_column: str | None = None
    use_keyword_rules: bool = True
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    min_text_length: int = 3

@dataclass
class ProcessingResult:
    input_rows: int
    valid_rows: int
    duplicate_rows: int
    review_rows: int
    labeled_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    output_dir: str

def _find_column(frame: pd.DataFrame, requested: str | None, aliases: tuple[str, ...], required: bool) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Column {requested!r} not found. Available columns: {list(frame.columns)}")
        return requested
    normalized = {str(c).strip().casefold(): str(c) for c in frame.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    if required:
        raise ValueError(f"Could not identify text column. Pass --text-column. Available: {list(frame.columns)}")
    return None

def _split(frame: pd.DataFrame, config: ProcessingConfig) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if frame.empty:
        empty = frame.copy()
        return empty, empty, empty
    train_r, val_r, test_r = config.train_ratio, config.validation_ratio, config.test_ratio
    if min(train_r,val_r,test_r) <= 0 or abs(train_r+val_r+test_r-1.0) > 1e-8:
        raise ValueError("train_ratio, validation_ratio and test_ratio must be positive and sum to 1.")
    y = frame["label_code"]
    stratify = y if y.value_counts().min() >= 3 and y.nunique() > 1 else None
    try:
        train, holdout = train_test_split(frame, test_size=val_r+test_r, random_state=config.random_state, stratify=stratify)
        holdout_stratify = holdout["label_code"] if stratify is not None and holdout["label_code"].value_counts().min() >= 2 else None
        val_fraction = val_r/(val_r+test_r)
        validation, test = train_test_split(holdout, test_size=1-val_fraction, random_state=config.random_state, stratify=holdout_stratify)
    except ValueError:
        # Small datasets cannot always be stratified; keep deterministic random splits.
        train, holdout = train_test_split(frame, test_size=val_r+test_r, random_state=config.random_state, stratify=None)
        validation, test = train_test_split(holdout, test_size=test_r/(val_r+test_r), random_state=config.random_state)
    return train.copy(), validation.copy(), test.copy()

def run_pipeline(config: ProcessingConfig) -> ProcessingResult:
    started = time.perf_counter()
    frame = read_source(config.input_path)
    input_rows = len(frame)
    text_col = _find_column(frame, config.text_column, TEXT_ALIASES, True)
    label_col = _find_column(frame, config.label_column, LABEL_ALIASES, False)
    data = pd.DataFrame()
    data["raw_content"] = frame[text_col].astype(str)
    data["cleaned_text"] = data["raw_content"].map(clean_text)
    data["normalized_text"] = data["cleaned_text"].map(normalize_text)
    data["source_file"] = Path(config.input_path).name
    if label_col:
        data["source_label"] = frame[label_col].astype(str)
    else:
        data["source_label"] = ""
    data = data[data["normalized_text"].str.len() >= config.min_text_length].copy()
    data = data.drop_duplicates(subset=["normalized_text"], keep="first").copy()
    data["label_code"] = None
    data["label_name"] = None
    data["label_source"] = "unlabeled"
    for idx, row in data.iterrows():
        code, name = canonical_label(row["source_label"])
        if code:
            data.at[idx,"label_code"], data.at[idx,"label_name"], data.at[idx,"label_source"] = code, name, "source"
        elif config.use_keyword_rules:
            code, name, source = keyword_label(row["cleaned_text"])
            if code:
                data.at[idx,"label_code"], data.at[idx,"label_name"], data.at[idx,"label_source"] = code, name, source
    data["needs_review"] = data["label_code"].isna()
    data["segmented_text"] = data["cleaned_text"].map(lambda s: " ".join(__import__("underthesea").word_tokenize(s, format="text").split()) if s else "")
    labeled = data[~data["needs_review"]].copy()
    review = data[data["needs_review"]].copy()
    train, validation, test = _split(labeled, config)
    for split_name, part in (("TRAIN",train),("VALIDATION",validation),("TEST",test)):
        part["split"] = split_name
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = ["raw_content","cleaned_text","normalized_text","segmented_text","label_code","label_name","label_source","source_file","split","needs_review"]
    for name, part in (("train.csv",train),("validation.csv",validation),("test.csv",test)):
        part.reindex(columns=columns).to_csv(output/name,index=False,encoding="utf-8-sig")
    review.reindex(columns=columns).to_csv(output/"needs_review.csv",index=False,encoding="utf-8-sig")
    labeled.reindex(columns=columns).to_csv(output/"labeled_all.csv",index=False,encoding="utf-8-sig")
    distribution = labeled.groupby(["label_code","label_name"],dropna=False).size().reset_index(name="count")
    distribution.to_csv(output/"label_distribution.csv",index=False,encoding="utf-8-sig")
    summary = {
        "input_file": str(config.input_path), "input_rows": input_rows,
        "valid_unique_rows": len(data), "duplicate_or_too_short_rows": input_rows-len(data),
        "labeled_rows": len(labeled), "needs_review_rows": len(review),
        "split_counts": {"train":len(train),"validation":len(validation),"test":len(test)},
        "split_ratios": {"train":config.train_ratio,"validation":config.validation_ratio,"test":config.test_ratio},
        "label_distribution": {str(k):int(v) for k,v in labeled["label_code"].value_counts().items()},
        "random_state":config.random_state,
        "processing_seconds":round(time.perf_counter()-started,3),
        "notes":["Exact duplicate removal uses normalized Vietnamese text and keeps the first row.",
                 "No fuzzy duplicate removal is performed to avoid deleting similar but distinct reports.",
                 "Rows without a reliable label are excluded from model splits and exported to needs_review.csv.",
                 "Splitting occurs after deduplication to prevent duplicate leakage across splits."]
    }
    (output/"dataset_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return ProcessingResult(input_rows,len(data),input_rows-len(data),len(review),len(labeled),len(train),len(validation),len(test),str(output))
