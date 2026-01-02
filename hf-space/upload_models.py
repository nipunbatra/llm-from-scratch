#!/usr/bin/env python3
"""
Upload trained models to HuggingFace Hub.

Usage:
    python upload_models.py

Requires:
    pip install huggingface_hub
    huggingface-cli login
"""

from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "nipunbatra/llm-from-scratch"
MODELS_DIR = "../models"

# Models to upload
MODEL_FILES = [
    "char_lm_names.pt",
    "transformer_shakespeare.pt",
    "instruction_tuned.pt",
    "dpo_aligned.pt",
    "python_code_lm.pt",
]


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository {REPO_ID} ready")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload each model
    for filename in MODEL_FILES:
        local_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(local_path):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"models/{filename}",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
        else:
            print(f"  Skipping {filename} (not found)")

    print("\nDone! Models available at:")
    print(f"  https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
