
from openai import OpenAI
import time

# Set your API key as an environment variable before running:
# export OPENAI_API_KEY="sk-..."

client = OpenAI()

# Upload (context managers so files close properly)
with open("data/train_SMALL.jsonl", "rb") as tf, open("data/val_SMALL.jsonl", "rb") as vf:
    train_file = client.files.create(file=tf, purpose="fine-tune")
    val_file   = client.files.create(file=vf, purpose="fine-tune")

job = client.fine_tuning.jobs.create(
    model="gpt-4o-2024-08-06",        # use the snapshot ID
    training_file=train_file.id,
    validation_file=val_file.id,
    suffix="tt-china-labels-v0",      # optional, helps you find it later
    hyperparameters={"n_epochs": 1}   # smoke test; bump for real runs
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
