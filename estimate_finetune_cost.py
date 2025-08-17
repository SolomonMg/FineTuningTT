"""
Estimates cost to fine tune labelled data, TikTok transcript & metadata.
    Also provides quick estimates of inference (prediction on new data).
Author: Sol Messing 
Input: train.jsonl, val.jsonl - jsonl files structured for
    OpenAI SFT. 
Output: summary stats estimating cost for SFT and inference. 
Usage below: 

Usage: 

python estimate_finetune_cost.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --epochs 2

Rates for gpt-4.1-mini-2025-04-14
python estimate_finetune_cost.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --train-rate-per-m 5\
    --infer-in-per-m .8\
    --infer-out-per-m 3.2\
    --est-infer-input-toks 435 \
    --est-infer-output-toks 25 \
    --est-infer-n-calls 400000


"""



#!/usr/bin/env python3
import argparse, json, math, os, sys
try:
    import tiktoken
except ModuleNotFoundError:
    sys.stderr.write("Please install tiktoken: pip install tiktoken\n")
    raise

ENCODING = "o200k_base"  # tokenizer used by GPT-4o

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_example_tokens(enc, example):
    """
    Estimate tokens by summing tokens of every messages[*].content.
    (Close approximation; actual training may add a tiny overhead per message.)
    """
    msgs = example.get("messages", [])
    total = 0
    for m in msgs:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(enc.encode(content))
        else:
            # If you ever switch to array-of-parts content, count each part's text
            try:
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(enc.encode(part["text"]))
            except Exception:
                pass
    return total

def human_millions(n):
    return f"{n/1_000_000:.3f}M"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to train JSONL")
    ap.add_argument("--val", default=None, help="optional path to val JSONL")
    ap.add_argument("--epochs", type=int, default=1, help="planned training epochs")
    # Pricing defaults (override these if OpenAI updates pricing)
    ap.add_argument("--train-rate-per-m", type=float, default=25.00,
                    help="training cost in $ per 1M tokens (default: 25.00 for GPT-4o)")
    ap.add_argument("--infer-in-per-m", type=float, default=5.00,
                    help="inference input $ per 1M tokens (default: 5.00 for GPT-4o)")
    ap.add_argument("--infer-out-per-m", type=float, default=20.00,
                    help="inference output $ per 1M tokens (default: 20.00 for GPT-4o)")
    # (Optional) quick inference sizing
    ap.add_argument("--est-infer-input-toks", type=int, default=None,
                    help="optional: expected avg input tokens per inference call")
    ap.add_argument("--est-infer-output-toks", type=int, default=None,
                    help="optional: expected avg output tokens per inference call")
    ap.add_argument("--est-infer-n-calls", type=int, default=None,
                    help="optional: number of planned inference calls")
    args = ap.parse_args()

    enc = tiktoken.get_encoding(ENCODING)

    # ----- Count tokens in train/val -----
    def tally(path):
        tot = 0
        n = 0
        min_ex = math.inf
        max_ex = 0
        for ex in iter_jsonl(path):
            t = count_example_tokens(enc, ex)
            tot += t
            n += 1
            min_ex = min(min_ex, t)
            max_ex = max(max_ex, t)
        avg = tot / max(n, 1)
        return {"examples": n, "tokens_total": tot, "avg_per_example": avg,
                "min_per_example": 0 if n == 0 else min_ex, "max_per_example": max_ex}

    train = tally(args.train)
    val = tally(args.val) if args.val else None

    # ----- Training cost -----
    train_tokens_total = train["tokens_total"]
    charged_training_tokens = train_tokens_total * max(args.epochs, 1)
    training_cost = (charged_training_tokens / 1_000_000.0) * args.train_rate_per_m

    # ----- Optional inference estimate -----
    infer_cost = None
    if args.est_infer_input_toks is not None and args.est_infer_output_toks is not None and args.est_infer_n_calls:
        in_tokens  = args.est_infer_input_toks  * args.est_infer_n_calls
        out_tokens = args.est_infer_output_toks * args.est_infer_n_calls
        cost_in  = (in_tokens  / 1_000_000.0) * args.infer_in_per_m
        cost_out = (out_tokens / 1_000_000.0) * args.infer_out_per_m
        infer_cost = {"calls": args.est_infer_n_calls,
                      "input_tokens": in_tokens, "output_tokens": out_tokens,
                      "cost_in": cost_in, "cost_out": cost_out, "cost_total": cost_in + cost_out}

    # ----- Print summary -----
    print("=== Token Tally ===")
    print(f"Train: {train['examples']} examples, {human_millions(train_tokens_total)} tokens "
          f"(avg {train['avg_per_example']:.0f}, min {train['min_per_example']}, max {train['max_per_example']})")
    if val:
        print(f"Val:   {val['examples']} examples, {human_millions(val['tokens_total'])} tokens "
              f"(avg {val['avg_per_example']:.0f})")
    print("\n=== Training Cost Estimate ===")
    print(f"Epochs: {args.epochs}")
    print(f"Charged training tokens (train_tokens × epochs): {human_millions(charged_training_tokens)}")
    print(f"Training rate: ${args.train_rate_per_m:.2f} per 1M tokens")
    print(f"=> Estimated training cost: ${training_cost:,.2f}")

    if infer_cost:
        print("\n=== Inference Cost Estimate (optional) ===")
        print(f"Calls: {infer_cost['calls']:,}")
        print(f"Input tokens total:  {human_millions(infer_cost['input_tokens'])}  @ ${args.infer_in_per_m:.2f}/1M = ${infer_cost['cost_in']:.2f}")
        print(f"Output tokens total: {human_millions(infer_cost['output_tokens'])} @ ${args.infer_out_per_m:.2f}/1M = ${infer_cost['cost_out']:.2f}")
        print(f"=> Estimated inference cost: ${infer_cost['cost_total']:.2f}")

if __name__ == "__main__":
    main()
