#!/usr/bin/env python3
"""
Batch download safetensors shards from HuggingFace with per-file resume and retry.

Supports:
  - Download all files or specific batches (--batch 1,3,5)
  - Per-file resume via HF hub's built-in range requests
  - Configurable retry with exponential backoff
  - Progress tracking with a local state file
  - Optional manifest decryption to show dataset-to-batch mapping (--show-map)

Usage:
  # List all batches and their sizes
  python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --list

  # Download batch 1 only
  python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --output ./data --batch 1

  # Download batches 1-3
  python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --output ./data --batch 1,2,3

  # Download everything
  python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --output ./data

  # Show which datasets map to which batches (requires password)
  python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --show-map --password YOUR_PASSWORD
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BATCH_SIZE_LIMIT = 5 * 1024 * 1024 * 1024  # 5 GB per batch


def get_repo_file_info(repo_id, token=None):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model", files_metadata=True, token=token)
    files = []
    for s in info.siblings:
        files.append({"name": s.rfilename, "size": s.size or 0})
    return files


def build_batches(shard_files):
    """Group shard files into ~5GB batches for manageable downloading."""
    shard_files.sort(key=lambda f: f["name"])
    batches = []
    current_batch = []
    current_size = 0

    for f in shard_files:
        if current_size + f["size"] > BATCH_SIZE_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(f)
        current_size += f["size"]

    if current_batch:
        batches.append(current_batch)

    return batches


def download_file_with_retry(repo_id, filename, local_dir, token=None, max_retries=5):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(1, max_retries + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                repo_type="model",
                token=token,
                resume_download=True,
            )
            return path
        except (HfHubHTTPError, ConnectionError, OSError, Exception) as e:
            if attempt == max_retries:
                print(f"    FAILED after {max_retries} attempts: {e}")
                return None
            wait = min(2 ** attempt, 60)
            print(f"    Retry {attempt}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)
    return None


def load_progress(progress_file):
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"completed": []}


def save_progress(progress_file, progress):
    progress_file.write_text(json.dumps(progress, indent=2))


def show_dataset_map(repo_id, password, token=None):
    """Download manifest.enc, decrypt it, and show dataset-to-shard mapping."""
    from huggingface_hub import hf_hub_download
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="manifest.enc",
            local_dir=tmpdir,
            repo_type="model",
            token=token,
        )
        enc_bytes = Path(path).read_bytes()

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    salt = enc_bytes[:16]
    nonce = enc_bytes[16:28]
    ciphertext = enc_bytes[28:]
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200_000)
    key = kdf.derive(password.encode("utf-8"))
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    manifest = json.loads(plaintext.decode("utf-8"))

    all_files = get_repo_file_info(repo_id, token)
    shard_files = [f for f in all_files if f["name"].startswith("shards/")]
    batches = build_batches(shard_files)

    shard_to_batch = {}
    for i, batch in enumerate(batches, 1):
        for f in batch:
            shard_to_batch[f["name"]] = i

    print(f"\n{'Dataset Name':<70} {'Shard Files':<45} {'Batch(es)'}")
    print("-" * 130)
    for ds in manifest["datasets"]:
        name = ds["yaml_name"]
        shards = ds["shard_files"]
        batch_ids = sorted(set(shard_to_batch.get(s, "?") for s in shards))
        batch_str = ",".join(str(b) for b in batch_ids)
        shard_str = ",".join(s.split("/")[-1].replace(".safetensors", "") for s in shards)
        if len(shard_str) > 42:
            shard_str = shard_str[:39] + "..."
        if len(name) > 68:
            name = name[:65] + "..."
        print(f"  {name:<68} {shard_str:<45} {batch_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch download safetensors shards with per-file resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. xtsssss/vision-encoder-v3.2-exp)")
    parser.add_argument("--output", default=None, help="Local directory to download into")
    parser.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--batch", default=None, help="Comma-separated batch numbers to download (e.g. 1,2,3)")
    parser.add_argument("--list", action="store_true", help="List batches and exit")
    parser.add_argument("--show-map", action="store_true", help="Show dataset-to-batch mapping (needs --password)")
    parser.add_argument("--password", default=None, help="Password for manifest decryption (only for --show-map)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per file (default: 5)")
    parser.add_argument("--mirror", default=None, help="HF mirror endpoint (e.g. https://hf-mirror.com)")
    args = parser.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Fetching file list from {args.repo}...")
    all_files = get_repo_file_info(args.repo, token)
    shard_files = [f for f in all_files if f["name"].startswith("shards/")]
    meta_files = [f for f in all_files if not f["name"].startswith("shards/")]

    batches = build_batches(shard_files)
    total_size = sum(f["size"] for f in shard_files)

    print(f"\nRepository: {args.repo}")
    print(f"Total shards: {len(shard_files)}, Total size: {total_size / 1e9:.1f} GB")
    print(f"Organized into {len(batches)} batches (~5GB each):\n")
    for i, batch in enumerate(batches, 1):
        batch_size = sum(f["size"] for f in batch)
        file_names = [f["name"].split("/")[-1] for f in batch]
        print(f"  Batch {i:2d}: {len(batch):2d} file(s), {batch_size / 1e9:.2f} GB")
        for fn in file_names:
            matching = [f for f in batch if f["name"].endswith(fn)]
            sz = matching[0]["size"] / 1e6 if matching else 0
            print(f"            - {fn} ({sz:.0f} MB)")

    if args.list:
        return

    if args.show_map:
        if not args.password:
            print("\nError: --show-map requires --password")
            sys.exit(1)
        show_dataset_map(args.repo, args.password, token)
        return

    if not args.output:
        print("\nError: --output is required for downloading")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_batches = None
    if args.batch:
        selected_batches = set(int(x.strip()) for x in args.batch.split(","))
        invalid = selected_batches - set(range(1, len(batches) + 1))
        if invalid:
            print(f"\nError: invalid batch number(s): {invalid}. Valid range: 1-{len(batches)}")
            sys.exit(1)

    progress_file = output_dir / ".download_progress.json"
    progress = load_progress(progress_file)
    completed_set = set(progress["completed"])

    files_to_download = []
    for f in meta_files:
        files_to_download.append(f)

    for i, batch in enumerate(batches, 1):
        if selected_batches and i not in selected_batches:
            continue
        files_to_download.extend(batch)

    new_files = [f for f in files_to_download if f["name"] not in completed_set]
    skip_count = len(files_to_download) - len(new_files)

    batch_label = f"batches {args.batch}" if args.batch else "all"
    dl_size = sum(f["size"] for f in new_files)
    print(f"\nDownloading {batch_label}: {len(new_files)} files ({dl_size / 1e9:.1f} GB)")
    if skip_count > 0:
        print(f"  (skipping {skip_count} already-completed files)")

    failed = []
    for idx, f in enumerate(new_files, 1):
        fname = f["name"]
        fsize = f["size"]
        print(f"\n[{idx}/{len(new_files)}] {fname} ({fsize / 1e6:.0f} MB)")

        result = download_file_with_retry(
            args.repo, fname, output_dir, token=token, max_retries=args.max_retries
        )
        if result:
            progress["completed"].append(fname)
            save_progress(progress_file, progress)
            print(f"    OK -> {result}")
        else:
            failed.append(fname)

    print(f"\n{'=' * 60}")
    print(f"Download summary:")
    print(f"  Completed: {len(new_files) - len(failed)}/{len(new_files)}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for fn in failed:
            print(f"    - {fn}")
        print(f"\nRe-run the same command to retry failed downloads.")
    else:
        print(f"  All files downloaded successfully!")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
