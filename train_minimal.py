#!/usr/bin/env python3
"""
Minimal training script for SFT via the OpenAI API.
Author: Sol Messing

Input: train.jsonl, val.jsonl — JSONL files structured for OpenAI SFT.
Output: Fine-tuned model hosted by OpenAI.

Usage:
  export OPENAI_API_KEY="sk-..."
  python train_minimal.py \
    --model gpt-4.1-mini-2025-04-14 \
    --train data/train_BAL_safe.jsonl \
    --val   data/val_BAL_safe.jsonl \
    --suffix tt-china-labels-v0 \
    --n-epochs 1 \
    --wait

Notes:
  - Use --wait to poll until the job finishes; omit it to exit after job creation.
  - Optional hyperparams: --batch-size, --lr-multiplier, --seed
"""

import argparse, os, time, sys
from openai import OpenAI

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-mini-2025-04-14",
                    help='Base model (e.g., "gpt-4.1-mini-2025-04-14", "gpt-4o-2024-08-06")')
    ap.add_argument("--train", required=True, help="Path to training JSONL")
    ap.add_argument("--val", required=True, help="Path to validation JSONL")
    ap.add_argument("--suffix", default="tt-china-labels-v0", help="Suffix to track the run")
    ap.add_argument("--n-epochs", type=int, default=1, help="Number of epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="Optional batch size")
    ap.add_argument("--lr-multiplier", type=float, default=None, help="Optional learning rate multiplier")
    ap.add_argument("--seed", type=int, default=None, help="Optional training seed")
    ap.add_argument("--wait", action="store_true", help="Poll until the job completes")
    ap.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls when --wait is set")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and exit without calling the API")
    return ap.parse_args()

def ensure_file(path, name):
    if not os.path.exists(path):
        sys.exit(f"{name} not found: {path}")
    if os.path.getsize(path) == 0:
        sys.exit(f"{name} is empty: {path}")

def main():
    args = parse_args()

    # Basic validation
    ensure_file(args.train, "Training file")
    ensure_file(args.val,   "Validation file")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    if args.dry_run:
        print("[dry-run] Inputs look good.")
        print(f"[dry-run] model={args.model} train={args.train} val={args.val} suffix={args.suffix}")
        return

    client = OpenAI()

    # Upload files
    with open(args.train, "rb") as tf, open(args.val, "rb") as vf:
        train_file = client.files.create(file=tf, purpose="fine-tune")
        val_file   = client.files.create(file=vf, purpose="fine-tune")

    # Build hyperparameters
    hparams = {"n_epochs": int(args.n_epochs)}
    if args.batch_size is not None:
        hparams["batch_size"] = int(args.batch_size)
    if args.lr_multiplier is not None:
        hparams["learning_rate_multiplier"] = float(args.lr_multiplier)
    if args.seed is not None:
        hparams["seed"] = int(args.seed)

    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        model=args.model,
        training_file=train_file.id,
        validation_file=val_file.id,
        suffix=args.suffix,
        hyperparameters=hparams
    )
    print("Fine-tune job ID:", job.id)
    print("Status:", job.status)

    if not args.wait:
        # Show a hint to fetch status later
        print("Use this to check later:\n  python - <<'PY'\nfrom openai import OpenAI\nc=OpenAI()\nprint(c.fine_tuning.jobs.retrieve('%s'))\nPY" % job.id)
        return

    # Poll until completion
    while True:
        j = client.fine_tuning.jobs.retrieve(job.id)
        status = j.status
        trained = getattr(j, "trained_tokens", None)
        print(f"status={status} trained_tokens={trained}")
        if status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(max(1, int(args.poll_interval)))

    if j.status != "succeeded":
        # Try to print a couple of events for debugging
        try:
            events = client.fine_tuning.jobs.list_events(job.id, limit=5)
            print("Last events:")
            for ev in events.data:
                print(" •", getattr(ev, "message", str(ev)))
        except Exception:
            pass
        sys.exit(f"Fine-tune did not succeed: {j.status}")

    print("Fine-tuned model:", j.fine_tuned_model)

if __name__ == "__main__":
    main()
