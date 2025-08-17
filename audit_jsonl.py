#!/usr/bin/env python3
"""
Quick audit for SFT JSONL exported by build_finetune_jsonl.py.
- Validates assistant label schema per example
- Prints distributions and joint crosstabs
- Checks language list rules (e.g., 'no_language' must be the only item)
- (Optional) estimates transcript length from the USER message block

Usage:
  python audit_jsonl.py --jsonl data/train_BAL.jsonl
  python audit_jsonl.py --jsonl data/val_BAL.jsonl --show-samples 3
"""
import argparse, json, re
from collections import Counter, defaultdict

ALLOWED_STANCE = {"good","bad","neutral","not_related","cannot_determine"}
ALLOWED_YNCD   = {"yes","no","cannot_determine"}
ALLOWED_LANG   = {"english","mandarin","spanish","other","no_language"}

def validate_labels(obj):
    if not isinstance(obj, dict):
        return "assistant content is not a JSON object"
    req = {"china_stance","china_sensitive","collective_action","languages"}
    if set(obj.keys()) != req:
        return f"unexpected keys: got {sorted(obj.keys())}, expected {sorted(req)}"
    if obj["china_stance"] not in ALLOWED_STANCE: return "bad china_stance"
    if obj["china_sensitive"] not in ALLOWED_YNCD: return "bad china_sensitive"
    if obj["collective_action"] not in ALLOWED_YNCD: return "bad collective_action"
    langs = obj["languages"]
    if not isinstance(langs, list): return "languages not a list"
    if any(l not in ALLOWED_LANG for l in langs): return "unknown language"
    if len(set(langs)) != len(langs): return "duplicate languages"
    if "no_language" in langs and len(langs) != 1: return "no_language must be only item"
    return None

def find_message(messages, role):
    for m in messages:
        if m.get("role") == role:
            return m.get("content","")
    return ""

TRANSCRIPT_RE = re.compile(r"Transcript:\s*(.*?)\nDescription:", re.DOTALL)

def transcript_len(user_block):
    if not user_block:
        return 0
    m = TRANSCRIPT_RE.search(user_block)
    if not m:
        return 0
    return len(m.group(1).strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--show-samples", type=int, default=0, help="print N sample assistant objects")
    args = ap.parse_args()

    n_total = n_valid = 0
    n_bad_schema = Counter()
    stance_ctr   = Counter()
    sens_ctr     = Counter()
    coll_ctr     = Counter()
    joint_ctr    = Counter()
    lang_ctr     = Counter()
    lang_combo   = Counter()
    tlen_sum = tlen_min = None
    tlen_max = 0

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_total += 1
            try:
                ex = json.loads(line)
            except Exception as e:
                n_bad_schema["not valid json line"] += 1
                continue

            msgs = ex.get("messages", [])
            user_txt = find_message(msgs, "user")
            assistant_txt = find_message(msgs, "assistant")

            # validate assistant label object
            try:
                labels = json.loads(assistant_txt)
            except Exception:
                n_bad_schema["assistant not json"] += 1
                continue

            err = validate_labels(labels)
            if err:
                n_bad_schema[err] += 1
                continue
            n_valid += 1

            # counts
            s = labels["china_stance"]
            y = labels["china_sensitive"]
            c = labels["collective_action"]
            stance_ctr[s] += 1
            sens_ctr[y]   += 1
            coll_ctr[c]   += 1
            joint_ctr[(s,y,c)] += 1

            # languages
            langs = labels["languages"]
            for l in langs:
                lang_ctr[l] += 1
            lang_combo[tuple(sorted(langs))] += 1

            # transcript length rough estimate (optional)
            tl = transcript_len(user_txt)
            tlen_sum = (tlen_sum or 0) + tl
            tlen_min = tl if tlen_min is None else min(tlen_min, tl)
            tlen_max = max(tlen_max, tl)

    # ----- report -----
    print(f"\nFile: {args.jsonl}")
    print(f"Total lines: {n_total}")
    print(f"Valid label objects: {n_valid} ({(n_valid/n_total*100):.1f}%)")
    if n_bad_schema:
        print("Schema issues:")
        for k,v in n_bad_schema.most_common():
            print(f"  {k:>32} : {v}")

    if n_valid == 0:
        return

    def show_counter(title, ctr, limit=None):
        print(f"\n{title}")
        for k,v in (ctr.most_common() if limit is None else ctr.most_common(limit)):
            print(f"  {str(k):>30} : {v}")
        print(f"  {'TOTAL':>30} : {sum(ctr.values())}")

    show_counter("china_stance distribution", stance_ctr)
    show_counter("china_sensitive distribution", sens_ctr)
    show_counter("collective_action distribution", coll_ctr)

    print("\nTop joint cells (stance | sensitive | collective):")
    for (s,y,c), n in joint_ctr.most_common(25):
        print(f"  {s:>16} | {y:>12} | {c:>14} : {n}")
    print(f"  TOTAL (all joints): {sum(joint_ctr.values())}")

    show_counter("Language tokens (flat counts)", lang_ctr)
    show_counter("Language combinations (top 15)", lang_combo, limit=15)

    if tlen_sum is not None:
        avg = tlen_sum / n_valid if n_valid else 0
        print(f"\nTranscript length (chars) — approx from USER block:")
        print(f"  min={tlen_min}  max={tlen_max}  mean={avg:.1f}")

    if args.show_samples > 0:
        print("\nSample assistant label objects:")
        shown = 0
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    labels = json.loads(find_message(ex.get("messages",[]), "assistant"))
                except Exception:
                    continue
                print(" ", labels)
                shown += 1
                if shown >= args.show_samples:
                    break

if __name__ == "__main__":
    main()
