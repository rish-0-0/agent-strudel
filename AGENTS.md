# Instructions for fine tuning

## 1. We need data first, so generate ample data using the training data generator agent

Run `uv run python training-data-gen/main.py`

## Once data is available in data/ directory

Run `uv run --project fine-tuner python fine-tuner/main.py`

(Smoke test: append `--epochs 1 --max-train-samples 4 --batch-size 2`)

## Generate WASM with the new model weights.

[will add command here]

Host your weights in a place of your choice

Run the model via WASM in the browser,

And watch a truly (in the browser) SLM agent generate music.