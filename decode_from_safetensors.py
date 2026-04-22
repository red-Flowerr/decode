#!/usr/bin/env python3
"""
Decode safetensors back to original parquet files.

Pipeline: safetensors -> uint8 tensor -> bytes -> zstd decompress -> parquet file
Manifest is AES-256-GCM decrypted with the provided password.

Supports partial decoding when only some shards have been downloaded.
Use --skip-missing to decode whatever is available without errors.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from safetensors.numpy import load_file


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=200_000,
    )
    return kdf.derive(password.encode("utf-8"))


def decrypt_manifest(enc_bytes: bytes, password: str) -> dict:
    salt = enc_bytes[:16]
    nonce = enc_bytes[16:28]
    ciphertext = enc_bytes[28:]
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    except Exception:
        raise ValueError("Decryption failed - wrong password or corrupted manifest")
    return json.loads(plaintext.decode("utf-8"))


def check_dataset_shards(repo_dir: Path, dataset_entry: dict):
    """Check how many shards are available for a dataset."""
    shard_files = dataset_entry["shard_files"]
    present = sum(1 for s in shard_files if (repo_dir / s).exists())
    return present, len(shard_files)


def decode_dataset(repo_dir: Path, dataset_entry: dict, output_dir: Path,
                   decompressor, skip_missing: bool = False,
                   path_separator: str = None):
    """Decode one dataset from safetensors shard(s) back to original files.

    Supports two manifest types:
      - Flat parquet datasets: tensor keys are plain filenames, output to <name>/data/
      - Directory trees: tensor keys use path_separator (e.g. '||') for nested paths
    """
    yaml_name = dataset_entry["yaml_name"]
    shard_files = dataset_entry["shard_files"]

    if path_separator:
        dataset_dir = output_dir / yaml_name
    else:
        dataset_dir = output_dir / yaml_name / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    total_bytes = 0
    missing_shards = []

    for shard_rel in shard_files:
        shard_path = repo_dir / shard_rel
        if not shard_path.exists():
            missing_shards.append(shard_rel)
            if skip_missing:
                print(f"  [SKIP] shard not found: {shard_path}")
                continue
            else:
                print(f"  [ERROR] shard not found: {shard_path}")
                print(f"         Use --skip-missing to decode available shards only")
                return 0, 0, missing_shards

        tensors = load_file(str(shard_path))
        for tensor_key, tensor in tensors.items():
            compressed_bytes = tensor.tobytes()
            raw_bytes = decompressor.decompress(compressed_bytes)

            if path_separator:
                rel_path = tensor_key.replace(path_separator, os.sep)
            else:
                rel_path = tensor_key

            out_file = dataset_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(raw_bytes)
            file_count += 1
            total_bytes += len(raw_bytes)

    return file_count, total_bytes, missing_shards


def main():
    parser = argparse.ArgumentParser(description="Decode safetensors back to parquet files")
    parser.add_argument("--repo-dir", required=True, help="Path to downloaded HF repo")
    parser.add_argument("--output", required=True, help="Output directory for restored parquet files")
    parser.add_argument("--password", required=True, help="Password for AES-256 manifest decryption")
    parser.add_argument("--filter", default=None,
                        help="Comma-separated yaml_names to decode (default: all)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip missing shards instead of stopping (for partial downloads)")
    parser.add_argument("--check", action="store_true",
                        help="Only check which datasets have all shards downloaded, don't decode")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output)

    enc_path = repo_dir / "manifest.enc"
    if not enc_path.exists():
        print(f"Error: manifest.enc not found in {repo_dir}")
        sys.exit(1)

    print("Decrypting manifest...")
    manifest = decrypt_manifest(enc_path.read_bytes(), args.password)
    datasets = manifest["datasets"]
    print(f"  Found {len(datasets)} datasets (encoding: {manifest['encoding']})")

    if args.filter:
        allowed = set(args.filter.split(","))
        datasets = [d for d in datasets if d["yaml_name"] in allowed]
        print(f"  Filtered to {len(datasets)} datasets")

    if args.check:
        print(f"\nShard availability check:")
        ready = 0
        partial = 0
        missing = 0
        for entry in datasets:
            present, total = check_dataset_shards(repo_dir, entry)
            status = "READY" if present == total else ("PARTIAL" if present > 0 else "MISSING")
            if status == "READY":
                ready += 1
            elif status == "PARTIAL":
                partial += 1
            else:
                missing += 1
            print(f"  [{status:7s}] {entry['yaml_name']} ({present}/{total} shards)")
        print(f"\nSummary: {ready} ready, {partial} partial, {missing} missing")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    decompressor = zstd.ZstdDecompressor()
    path_separator = manifest.get("path_separator")
    manifest_type = manifest.get("type", "flat")
    if path_separator:
        print(f"  Manifest type: directory_tree (separator: '{path_separator}')")

    total_files = 0
    total_size = 0
    all_missing = []

    for i, entry in enumerate(datasets, 1):
        name = entry["yaml_name"]
        present, total = check_dataset_shards(repo_dir, entry)
        if present == 0 and args.skip_missing:
            print(f"[{i}/{len(datasets)}] {name} - no shards available, skipping")
            continue
        print(f"[{i}/{len(datasets)}] {name} ({present}/{total} shards)")
        fc, tb, missing = decode_dataset(
            repo_dir, entry, output_dir, decompressor,
            skip_missing=args.skip_missing, path_separator=path_separator
        )
        total_files += fc
        total_size += tb
        all_missing.extend(missing)
        print(f"    {fc} files, {tb / 1e6:.1f} MB")

    print(f"\nDone! Restored {total_files} files, {total_size / 1e6:.1f} MB total")
    print(f"  Output dir: {output_dir}")

    if all_missing:
        print(f"\n  WARNING: {len(all_missing)} shard(s) were missing.")
        print(f"  Download them first, then re-run to decode remaining datasets.")

    mapping = {}
    for entry in datasets:
        ds_dir = output_dir / entry["yaml_name"] / "data"
        if ds_dir.exists() and any(ds_dir.iterdir()):
            mapping[entry["yaml_name"]] = str(ds_dir)
    mapping_path = output_dir / "path_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"  Path mapping: {mapping_path}")


if __name__ == "__main__":
    main()
