"""CLI entry point: python ml/scripts/prepare_dataset.py --input ..."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"ml"))
from src.data_processing.pipeline import ProcessingConfig,run_pipeline

def main() -> int:
    parser=argparse.ArgumentParser(description="Clean, normalize, deduplicate, label and split ST-NLP report data.")
    parser.add_argument("--input",required=True,help="CSV, TSV, XLSX, JSON, JSONL or TXT source file.")
    parser.add_argument("--output",default="ml/data/processed")
    parser.add_argument("--text-column",default=None)
    parser.add_argument("--label-column",default=None)
    parser.add_argument("--no-keyword-rules",action="store_true",help="Disable weak keyword labeling; keep unknown labels for review.")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--min-text-length",type=int,default=3)
    args=parser.parse_args()
    config=ProcessingConfig(input_path=args.input,output_dir=args.output,text_column=args.text_column,label_column=args.label_column,use_keyword_rules=not args.no_keyword_rules,random_state=args.seed,min_text_length=args.min_text_length)
    result=run_pipeline(config)
    print(json.dumps(result.__dict__,ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
