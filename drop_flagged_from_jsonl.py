#!/usr/bin/env python3
"""
Drop JSONL rows by 1-based line_number from a flagged CSV.

Usage:
  python drop_flagged_from_jsonl.py \
    --in data/train_BAL_rewritten_v7.jsonl \
    --flagged data/flagged_after_v7.csv \
    --out data/train_BAL_final.jsonl \
    --write-dropped data/train_BAL_final_dropped.jsonl
"""
import argparse, csv, os

def read_flagged_lines(csv_path):
    to_drop = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "line_number" not in r.fieldnames:
            raise SystemExit(f"{csv_path}: missing required column 'line_number'")
        for row in r:
            s = (row.get("line_number") or "").strip()
            if not s:
                continue
            try:
                ln = int(s)
                if ln > 0:
                    to_drop.add(ln)
            except ValueError:
                pass
    return to_drop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL")
    ap.add_argument("--flagged", required=True, nargs="+", help="flagged CSV(s) with line_number")
    ap.add_argument("--out", dest="outp", required=True, help="output JSONL (flagged lines removed)")
    ap.add_argument("--write-dropped", dest="write_dropped", default=None,
                    help="optional JSONL with the dropped lines")
    args = ap.parse_args()

    to_drop = set()
    for p in args.flagged:
        to_drop |= read_flagged_lines(p)

    os.makedirs(os.path.dirname(args.outp) or ".", exist_ok=True)
    dropped_fh = open(args.write_dropped, "w", encoding="utf-8") if args.write_dropped else None

    kept = dropped = total = 0
    with open(args.inp, "r", encoding="utf-8") as fi, open(args.outp, "w", encoding="utf-8") as fo:
        for lineno, line in enumerate(fi, start=1):
            total += 1
            if lineno in to_drop:
                dropped += 1
                if dropped_fh:
                    dropped_fh.write(line)
                continue
            fo.write(line)
            kept += 1

    if dropped_fh:
        dropped_fh.close()

    print(f"Flagged line numbers: {len(to_drop)}")
    print(f"Processed {total} lines → kept {kept}, dropped {dropped}")
    print(f"Output → {args.outp}")
    if args.write_dropped:
        print(f"Dropped → {args.write_dropped}")

if __name__ == "__main__":
    main()
