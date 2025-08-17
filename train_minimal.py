"""
Minimal training script for SFT via the OpenAI API. Initial training
    on gpt-4o-2024-08-06, though this can be changed. 
Author: Sol Messing 
Input: train.jsonl, val.jsonl - jsonl files structured for
    OpenAI SFT. 
Output: fine-tuned model at OpenAI

"""

from openai import OpenAI
import time

OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
# ALTs: "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"

MODEL_SUFFIX = "tt-china-labels-v0"
N_EPOCHS = 1

# Set your API key as an environment variable before running:
# export OPENAI_API_KEY="sk-..."

client = OpenAI()

# Upload (context managers so files close properly)
with open("data/train.jsonl", "rb") as tf, open("data/SMALL.jsonl", "rb") as vf:
    train_file = client.files.create(file=tf, purpose="fine-tune")
    val_file   = client.files.create(file=vf, purpose="fine-tune")

job = client.fine_tuning.jobs.create(
    model=OPENAI_MODEL,        # use the snapshot ID
    training_file=train_file.id,
    validation_file=val_file.id,
    suffix=MODEL_SUFFIX,      # optional, helps you find it later
    hyperparameters={"n_epochs": N_EPOCHS}   # smoke test; bump for real runs
)
print("Fine-tune job ID:", job.id)

# Poll until done (or use list_events)
while True:
    j = client.fine_tuning.jobs.retrieve(job.id)
    print(f"status={j.status} trained_tokens={getattr(j, 'trained_tokens', None)}")
    if j.status in ("succeeded", "failed", "cancelled"):
        break
    time.sleep(10)

if j.status != "succeeded":
    raise SystemExit(f"Fine-tune did not succeed: {j.status}")

print("Fine-tuned model:", j.fine_tuned_model)
