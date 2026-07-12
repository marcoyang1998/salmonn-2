# SALMONN-2

This repository contains the supported training and inference code for SALMONN-2:

```text
audio -> 128-bin filterbank -> Zipformer2 -> MLP connector -> Qwen3
```

It intentionally excludes datasets, data-generation pipelines, benchmark runners, scoring code,
experimental encoders, the unused reasoning network, and pause embeddings.

## Environment setup

The following instructions create a clean environment from scratch. Python 3.10 or 3.11 is
recommended.

### 1. Create and activate an environment

Using Conda:

```bash
conda create -n salmonn2 python=3.10 -y
conda activate salmonn2
python -m pip install --upgrade pip setuptools wheel
```

Alternatively, using Python's built-in virtual environments:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch for your platform

Install PyTorch and TorchAudio using the command appropriate for your operating system, accelerator,
and driver. Select the command at <https://pytorch.org/get-started/locally/>. SALMONN-2 does not
hard-code a CUDA version.

For example, a CPU-only installation can be created with:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA or AMD GPU, use the CUDA or ROCm command supplied by the PyTorch selector instead.
Keep `torch` and `torchaudio` on matching releases.

Confirm that PyTorch can see the intended device:

```bash
python -c "import torch, torchaudio; print('torch:', torch.__version__); print('torchaudio:', torchaudio.__version__); print('CUDA available:', torch.cuda.is_available())"
```

CPU installation is useful for imports and basic validation, but inference with the released 8B
model normally requires one or more GPUs with sufficient aggregate memory.

### 3. Install SALMONN-2

From the repository root:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

The first command installs the common runtime dependencies. The second installs this repository in
editable mode without asking pip to reconsider the platform-specific PyTorch installation.

For training, install the additional dependencies:

```bash
pip install 'accelerate>=1.0' 'deepspeed>=0.18'
```

### 4. Optional: install FlashAttention

FlashAttention is not required. The portable default is PyTorch scaled-dot-product attention
(`sdpa`). To use it, set this in the training configuration:

```json
"attn_implementation": "sdpa"
```

On a supported NVIDIA system, FlashAttention may provide better speed and memory use. Install it
only after PyTorch is working:

```bash
pip install packaging psutil ninja
MAX_JOBS=4 pip install 'flash-attn>=2.7' --no-build-isolation
```

Then select it in the configuration:

```json
"attn_implementation": "flash_attention_2"
```

FlashAttention compilation requires a compatible GPU, CUDA toolkit, compiler, and sufficient host
memory. If installation fails, continue with `sdpa`.

### 5. Verify the installation

```bash
python -m compileall -q salmonn scripts
python -c "import torch, torchaudio, transformers, peft, lhotse; import salmonn; from salmonn import AudioProcessor; AudioProcessor(); print('SALMONN-2 environment is ready')"
python scripts/infer.py --help
python scripts/infer_batch.py --help
python scripts/train.py --help
```

To run the repository tests, install the test extra and run Pytest:

```bash
pip install 'pytest>=8'
pytest -q
```

The environment setup installs code dependencies only. Inference additionally requires the
released SALMONN-2 checkpoint. Training from scratch requires Qwen3 weights, a pretrained
Zipformer2 checkpoint, a prepared manifest, and the referenced audio files.

## Manifest format

Training consumes an already prepared JSON conversation manifest. Every `<audio>` placeholder
must correspond, in order, to one entry in `audios`.

```json
[
  {
    "audios": ["/path/to/audio.wav"],
    "messages": [
      {"role": "user", "content": "<audio>Please describe this audio."},
      {"role": "assistant", "content": "A person speaks over music."}
    ]
  }
]
```

The loader resamples audio to 16 kHz and computes 128-bin filterbanks. Dataset downloading,
conversion, augmentation, and task-specific prompting are outside this repository.

## Train

Edit the model paths in `configs/train_stage1.json`, then run:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/train_stage1.json \
  --data_path /path/to/train.json \
  --output_dir output/stage1
```

Stage two loads the stage-one checkpoint and applies Qwen3 LoRA. Update `model_name_or_path` in
`configs/train_stage2.json` before launching it in the same way. DeepSpeed can be enabled by adding
`"deepspeed": "configs/deepspeed_zero2.json"` to the `training` block.

## Inference

Converted checkpoints use the standard Hugging Face auto classes with bundled custom model code.
Pass `trust_remote_code=True` when loading them. A complete Python example is available in
[`examples/inference_hf.py`](examples/inference_hf.py).

```bash
python examples/inference_hf.py \
  --model_path /path/to/salmonn-2-hf \
  --audio example.wav \
  --prompt "Please describe the audio."
```

The equivalent Python loading pattern is:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()
model.register_nl_timestamp_tokenizer(tokenizer)
```

The repository CLI uses the same Hugging Face loading path:

```bash
python scripts/infer.py \
  --model_path /path/to/salmonn-2-checkpoint \
  --audio example.wav \
  --prompt "Please describe the audio."
```

The example and inference CLIs remove the literal `<think>` and `</think>` boundary tags from
displayed responses. This is output formatting only and does not alter generation or token IDs.

Repeat `--audio` for prompts containing multiple audio inputs. For JSON batch generation:

```bash
python scripts/infer_batch.py \
  --model_path /path/to/salmonn-2-checkpoint \
  --input examples/inference_manifest.json \
  --output predictions.jsonl
```

The batch command only generates responses; it does not compute benchmark scores.

## Convert a training checkpoint

Legacy Trainer/DeepSpeed checkpoints must be converted before they can be loaded by the release
model. The converter merges Qwen LoRA adapters, rewrites legacy PEFT parameter names, creates a
SALMONN-2 Hugging Face configuration, bundles the custom model code, and excludes optimizer and
other training-only state.

```bash
python scripts/convert_checkpoint.py \
  --input /path/to/experiment/checkpoint-50000 \
  --output /path/to/salmonn-2-hf
```

By default, the converter reads `model_args` from `INPUT/../config.json`. Specify a different file
with `--training-config` when necessary. The output directory must be new or empty. Conversion is
performed one safetensor shard at a time on CPU; it does not require a GPU, but it needs enough RAM
for the largest input shard.

Always compare the legacy and converted models on a fixed set of audio prompts before publishing
the converted checkpoint.

## Checkpoints

The Zipformer checkpoint may either be a raw state dictionary or contain its state dictionary
under the `model` key. Full SALMONN-2 checkpoints use the Hugging Face `save_pretrained` layout.

The Zipformer source files retain their upstream copyright and Apache-2.0 notices. Confirm the
license and distribution terms for Qwen3 and released model weights separately.
