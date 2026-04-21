#!/usr/bin/env python3
"""
Decode safetensors back to original parquet files.

Pipeline: safetensors -> uint8 tensor -> bytes -> zstd decompress -> parquet file
Manifest is AES-256-GCM decrypted with the provided password.
"""

import argparse
import json
import os
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


def decode_dataset(repo_dir: Path, dataset_entry: dict, output_dir: Path, decompressor):
    """Decode one dataset from safetensors shard(s) back to parquet files."""
    yaml_name = dataset_entry["yaml_name"]
    shard_files = dataset_entry["shard_files"]
    dataset_dir = output_dir / yaml_name / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    total_bytes = 0

    for shard_rel in shard_files:
        shard_path = repo_dir / shard_rel
        if not shard_path.exists():
            print(f"  [WARN] shard not found: {shard_path}, skipping")
            continue

        tensors = load_file(str(shard_path))
        for filename, tensor in tensors.items():
            compressed_bytes = tensor.tobytes()
            raw_bytes = decompressor.decompress(compressed_bytes)
            out_file = dataset_dir / filename
            out_file.write_bytes(raw_bytes)
            file_count += 1
            total_bytes += len(raw_bytes)

    return file_count, total_bytes


def main():
    parser = argparse.ArgumentParser(description="Decode safetensors back to parquet files")
    parser.add_argument("--repo-dir", required=True, help="Path to downloaded HF repo")
    parser.add_argument("--output", required=True, help="Output directory for restored parquet files")
    parser.add_argument("--password", required=True, help="Password for AES-256 manifest decryption")
    parser.add_argument("--filter", default=None, help="Comma-separated yaml_names to decode (default: all)")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    decompressor = zstd.ZstdDecompressor()
    total_files = 0
    total_size = 0

    for i, entry in enumerate(datasets, 1):
        name = entry["yaml_name"]
        print(f"[{i}/{len(datasets)}] {name}")
        fc, tb = decode_dataset(repo_dir, entry, output_dir, decompressor)
        total_files += fc
        total_size += tb
        print(f"    {fc} files, {tb / 1e6:.1f} MB")

    print(f"\nDone! Restored {total_files} files, {total_size / 1e6:.1f} MB total")
    print(f"  Output dir: {output_dir}")

    # Write a mapping file for convenience
    mapping = {}
    for entry in manifest["datasets"]:
        mapping[entry["yaml_name"]] = str(output_dir / entry["yaml_name"] / "data")
    mapping_path = output_dir / "path_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"  Path mapping: {mapping_path}")


if __name__ == "__main__":
    import sys
    main()
