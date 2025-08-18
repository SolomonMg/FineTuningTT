#!/usr/bin/env python3
"""
Check a fine-tuning JSONL with OpenAI Moderation and output ONLY flagged rows.
Adds throttling, retries, and hashing cache to avoid 429s.

Usage:
  export OPENAI_API_KEY="sk-..."
  python check_moderation.py \
    --in data/train_BAL.jsonl \
    --out-csv data/flagged.csv \
    --cache .modcache_moderation.json \
    --batch-size 8 --rpm 30 --max-rows 1000
"""
import argparse, csv, json, os, sys, time, hashlib, random
from typing import Dict
import openai
from openai import OpenAI

def find_user_text(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content","")
    return ""

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_cache(path:str)->Dict[str,dict]:
    if not path or not os.path.exists(path): return {}
    try:
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(path:str, cache:dict):
    if not path: return
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(cache,f)
    os.replace(tmp, path)

def call_with_retry(client, texts, max_retries=6, base_backoff=1.0):
    for attempt in range(max_retries):
        try:
            return client.moderations.create(model="omni-moderation-latest", input=texts)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIStatusError) as e:
            # 429 or transient 5xx
            sleep = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep)
            if attempt == max_retries - 1:
                raise
        except Exception:
            # unknown; brief backoff then rethrow at last attempt
            sleep = 0.5 + random.uniform(0, 0.5)
            time.sleep(sleep)
            if attempt == max_retries - 1:
                raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=30.0, help="requests per minute throttle")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--cache", default=None, help="JSON cache keyed by sha256(user_text)")
    args = ap.parse_args()

    if not args.out_csv and not args.out_jsonl:
        sys.exit("Specify --out-csv or --out-jsonl (or both).")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    # Prepare outputs
    csvw = None; csvf = None
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        csvf = open(args.out_csv, "w", newline="", encoding="utf-8")
        csvw = csv.writer(csvf); csvw.writerow(["line_number","categories","user_excerpt"])
    jfw = None; jff = None
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        jff = open(args.out_jsonl, "w", encoding="utf-8"); jfw = jff

    # Read JSONL
    items = []  # (line_no, raw_obj, user_text, hash)
    with open(args.inp, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if args.max_rows and ln > args.max_rows: break
            s = line.strip()
            if not s: continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            msgs = obj.get("messages", [])
            user_text = find_user_text(msgs)
            if user_text is None: user_text = ""
            items.append((ln, obj, user_text, sha(user_text)))

    if not items:
        print("No items with a user message found.")
        return

    client = OpenAI()
    cache = load_cache(args.cache)
    flagged_cnt = 0
    req_interval = 60.0 / max(1.0, args.rpm)
    last_req_time = 0.0

    # Process in batches, using cache where possible
    total = len(items)
    for i in range(0, total, args.batch_size):
        chunk = items[i:i+args.batch_size]
        # Prepare batch: only texts not in cache
        to_query_idx = []
        batch_texts = []
        for j, (_, _, utext, h) in enumerate(chunk):
            if h in cache:
                continue
            to_query_idx.append(j)
            batch_texts.append(utext)

        # Throttle requests per minute (one request per batch)
        if batch_texts:
            now = time.time()
            elapsed = now - last_req_time
            if elapsed < req_interval:
                time.sleep(req_interval - elapsed)
            resp = call_with_retry(client, batch_texts)
            last_req_time = time.time()
            # Store in cache
            for j, res in zip(to_query_idx, resp.results):
                h = chunk[j][3]
                cache[h] = res.model_dump()
            save_cache(args.cache, cache)

        # Write flagged outputs
        for (ln, obj, utext, h) in chunk:
            res = cache.get(h)
            if not res:
                continue  # shouldn’t happen
            flagged = bool(res.get("flagged", False))
            if not flagged:
                continue
            cats = [k for k,v in (res.get("categories") or {}).items() if v]
            excerpt = (utext[:200] + "…") if len(utext) > 200 else utext
            if csvw:
                csvw.writerow([ln, ";".join(sorted(cats)), excerpt])
            if jfw:
                out = {"line_number": ln, "moderation": {"flagged": True, "categories": sorted(cats)}, "record": obj}
                jfw.write(json.dumps(out, ensure_ascii=False) + "\n")
            flagged_cnt += 1

        # light progress
        done = min(i + args.batch_size, total)
        print(f"[check] {done}/{total} processed; flagged so far: {flagged_cnt}", flush=True)

    if csvf: csvf.close()
    if jff:  jff.close()
    print(f"\nDone. Scanned: {total}  Flagged: {flagged_cnt}")
    if args.out_csv:   print(f"Flagged → CSV:   {args.out_csv}")
    if args.out_jsonl: print(f"Flagged → JSONL: {args.out_jsonl}")

if __name__ == "__main__":
    main()
