# Vision Encoder Weights Decoder

Decode safetensors model weights back to original format.  
Supports batch downloading with per-file resume for large repositories.

## Install

```bash
pip install safetensors zstandard numpy huggingface_hub cryptography
```

## Quick Start

### 1. List available batches

```bash
python batch_download.py --repo xtsssss/vision-encoder-v3.2-exp --list
```

### 2. Download in batches (~5GB each)

```bash
# Download batch 1 only
python batch_download.py \
  --repo xtsssss/vision-encoder-v3.2-exp \
  --output ./model_data \
  --token YOUR_HF_TOKEN \
  --batch 1

# Download batches 1,2,3
python batch_download.py \
  --repo xtsssss/vision-encoder-v3.2-exp \
  --output ./model_data \
  --token YOUR_HF_TOKEN \
  --batch 1,2,3

# Download all (re-run to resume if interrupted)
python batch_download.py \
  --repo xtsssss/vision-encoder-v3.2-exp \
  --output ./model_data \
  --token YOUR_HF_TOKEN
```

### 3. Check download status

```bash
python decode_from_safetensors.py \
  --repo-dir ./model_data \
  --output ./output \
  --password "<password>" \
  --check
```

### 4. Decode (supports partial downloads)

```bash
# Decode all available datasets
python decode_from_safetensors.py \
  --repo-dir ./model_data \
  --output ./output_data \
  --password "<password>" \
  --skip-missing

# Decode specific datasets only
python decode_from_safetensors.py \
  --repo-dir ./model_data \
  --output ./output_data \
  --password "<password>" \
  --filter code_switch,artifacts_1209
```

## Options

### batch_download.py

| Flag | Description |
|------|-------------|
| `--repo` | HuggingFace repo ID |
| `--output` | Local download directory |
| `--token` | HF token (or set `HF_TOKEN` env) |
| `--batch` | Comma-separated batch numbers (e.g. `1,2,3`) |
| `--list` | List all batches and exit |
| `--show-map` | Show dataset-to-batch mapping (needs `--password`) |
| `--mirror` | HF mirror endpoint (e.g. `https://hf-mirror.com`) |
| `--max-retries` | Max retries per file (default: 5) |

### decode_from_safetensors.py

| Flag | Description |
|------|-------------|
| `--repo-dir` | Path to downloaded data |
| `--output` | Output directory for restored files |
| `--password` | Decryption password |
| `--filter` | Comma-separated dataset names to decode |
| `--skip-missing` | Skip missing shards (for partial downloads) |
| `--check` | Check shard availability without decoding |

## Tips

- **Resume**: Re-run `batch_download.py` with the same `--output` to resume. Already-downloaded files are skipped automatically.
- **Mirror**: Use `--mirror https://hf-mirror.com` if direct HF access is slow.
- **Partial decode**: Download a few batches, use `--check` to see which datasets are ready, then decode with `--skip-missing`.
