# Question Generation and Evaluation

A framework for training and evaluating question-answering models on Hebrew text, with support for both local models and API-based evaluation using GPT and Gemini.

## Description

This project provides tools for:
- Fine-tuning language models for question-answering tasks using LoRA (Low-Rank Adaptation)
- Evaluating model performance on question-answering datasets
- Running inference with support for quantization and vLLM acceleration
- Benchmarking against commercial models (GPT, Gemini)

## Installation

### Using UV (Recommended)

```bash
uv sync
```

### Using pip

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install accelerate bitsandbytes flashinfer-cubin flashinfer-python \
  google-genai liger-kernel "numpy==2.2" nvidia-modelopt openai optimum \
  peft pillow polars scikit-learn torch transformers "trl[vllm]"
```

## Usage

### Training Models

Fine-tune a model using the training script:

```bash
python fine_tune_qg.py \
  --train-data-path data/train.csv \
  --dev-data-path data/dev.csv \
  --prompt-path prompts/training_prompt.txt \
  --save-path models/my-model \
  --model google/gemma-2-9b-it \
  --batch-size 4 \
  --steps 512 \
  --learning-rate 5e-4
```

**Optional flags:**
- `--use-8bit`: Enable 8-bit quantization
- `--use-4bit`: Enable 4-bit quantization
- `--device {cpu,cuda,auto}`: Specify device for training

### Running Inference

Evaluate a model on a question-answering dataset:

```bash
python main.py \
  --config-path config.toml \
  --data-path data/test.csv \
  --save-path results/answers.csv \
  --batch-size 8
```

**Configuration file (config.toml) example:**
```toml
model_name = "google/gemma-2-9b-it"
prompt_path = "prompts/eval_prompt.txt"
```

**Optional flags:**
- `--use-vllm`: Use vLLM for faster inference
- `--use-8bit`: Enable 8-bit quantization
- `--use-4bit`: Enable 4-bit quantization
- `--quantization {bitsandbytes,modelopt,fp8}`: Quantization method (requires --use-vllm)
- `--device {cpu,cuda,auto}`: Specify device
- `--num-of-shots N`: Enable few-shot prompting with N examples

### Running Evaluation Scripts

#### GPT Evaluation

Evaluate using OpenAI's GPT models via batch API:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_ORG="your-org-id"
export OPENAI_PROJECT="your-project-id"

python gpt_eval.py \
  --prompt-path prompts/eval_prompt.txt \
  --data-path data/test.csv \
  --model gpt-5-nano-2025-08-07
```

**Available models:**
- `gpt-5-nano-2025-08-07`
- `gpt-5-mini-2025-08-07`
- `gpt-5-2025-08-07`

#### Gemini Evaluation

Evaluate using Google's Gemini models:

```bash
export GEMINI_API_KEY="your-api-key"

python gemini_eval.py \
  --prompt-path prompts/eval_prompt.txt \
  --data-path data/test.csv \
  --save-path results/gemini_results.csv \
  --model gemini-2.5-flash
```

**Available models:**
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

## Data Format

Input CSV files should contain the following columns:
- `context`: The passage/context text
- `question`: The question to answer
- `answer`: The expected answer (for training/evaluation)
- `is_impossible`: Binary flag (1 if question is not answerable, 0 otherwise)

## License

See [LICENSE](LICENSE) file for details.
