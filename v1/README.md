# 🚀 QLoRA Multi-GPU Fine-Tuning with Unsloth (Qwen-7B)

This project provides a **production-grade pipeline** to fine-tune a **Qwen-7B model using QLoRA with Unsloth**, fully optimized for **multi-GPU training** using **Hugging Face Accelerate**.

## 📦 Project Structure

```
qlora_multigpu_project/
├── config.py             # All training parameters and config settings
├── data.py               # Dataset loading and tokenization logic
├── train.py              # Main training loop with Accelerate
├── utils.py              # Logging, checkpointing, and eval tools
├── requirements.txt      # List of dependencies
└── README.md             # This file
```

## ✅ Key Features

- ✅ Fine-tunes **Qwen-7B** using **QLoRA** via `unsloth`.
- ✅ Configurable **multi-GPU training** with `accelerate`.
- ✅ **Mid-size dataset** support (~10K–100K samples).
- ✅ Modular code with centralized `Config` class.
- ✅ Logging, evaluation, and checkpointing included.
- ✅ Uses `torch.bfloat16` or `fp16` for optimized performance.
- ✅ Easily extendable for other models/datasets.

## ⚙️ Prerequisites

- Python 3.10+
- 2 or more GPUs with ≥24GB VRAM (e.g., T4, A100)
- CUDA-compatible environment
- Linux or WSL recommended

## 📥 Installation

1. Clone the repo:

```bash
git clone https://github.com/your-org/qlora_multigpu_project.git
cd qlora_multigpu_project
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚡ Setup Accelerate

First, configure `accelerate`:

```bash
accelerate config
```

Recommended settings:
- Compute environment: **Multi-GPU**
- Mixed precision: **bf16** (or fp16 if unsupported)
- Use DeepSpeed: **No** (unless enabled manually)
- Use torch.compile: **Yes**

Then launch with:

```bash
accelerate launch train.py
```

## 🛠 Configuration

Edit `config.py` to adjust parameters like:
- `model_name` (e.g., `"unsloth/qwen-7b-qlora"`)
- `dataset_name` (e.g., `"timdettmers/openassistant-guanaco"`)
- `batch_size`, `learning_rate`, `gradient_accumulation_steps`, etc.

## 📊 Monitoring

- Use `nvidia-smi` or `accelerate env` to check GPU utilization.
- Logs will display eval loss and training stats per step.
- Add `wandb` or `tensorboard` in `utils.py` to integrate full training dashboards.

## 🧪 Evaluation

The script evaluates loss during training every N steps. Optionally, configure:
```python
eval_steps = 200
save_best_model = True
```

## 📦 Output

- Fine-tuned model checkpoint: `./checkpoints/best_model`
- Tokenizer + training logs
- Config snapshot for reproducibility

## 🧠 Tips

- Avoid `device_map="auto"`; use Accelerate to **truly parallelize** workloads.
- If GPUs are still underutilized, tune:
  - `gradient_accumulation_steps`
  - `per_device_train_batch_size`
  - Use smaller `max_seq_length` to reduce VRAM pressure.

## 📌 Notes

- Compatible with Hugging Face Transformers ≥4.38 and Unsloth latest release.
- Requires internet access for downloading base models & datasets unless pre-downloaded.

## 🤝 Acknowledgements

- [Unsloth by Hugging Face](https://github.com/unslothai/unsloth)
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate)
- [Qwen Model Series](https://huggingface.co/models)