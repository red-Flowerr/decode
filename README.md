# Vision Encoder Weights Decoder

Decode safetensors model weights back to original format.

## Install

```bash
pip install safetensors zstandard numpy huggingface_hub cryptography
```

## Usage

```bash
# 1. Download model weights
huggingface-cli download xtsssss/vision-encoder-v3.2-exp --local-dir ./model_data

# 2. Decode
python decode_from_safetensors.py \
  --repo-dir ./model_data \
  --output ./output_data \
  --password "<password>"
```

## Options

- `--filter`: Comma-separated names to selectively decode specific shards
