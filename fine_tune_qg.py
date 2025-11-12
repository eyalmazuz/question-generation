import argparse

import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def preprocess_function(example, prompt):
    user_input = {
        "role": "user",
        "content": prompt.format(
            question=example["question"],
            context=example["context"],
        ),
    }
    answer = "Not answerable" if example["is_impossible"] == 1 else example["answer"]
    return {
        "prompt": user_input,
        "completion": [{"role": "assistant", "content": f"{answer}"}],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path", type=str, required=True, help="Path to training data."
    )
    parser.add_argument(
        "--dev-data-path", type=str, required=True, help="Path to validation data."
    )
    parser.add_argument(
        "--prompt-path", type=str, required=True, help="Path to training prompt."
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to save the trained model."
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2-9b-it", help="model to fine-tune."
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Whether to use 8bit mix-precision or not.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Whether to use 8bit mix-precision or not.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        help="Which device to use when running the model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size to use during training."
    )
    parser.add_argument(
        "--steps", type=int, default=512, help="Number of training steps."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Training learning rate."
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    with open(args.prompt_path, "r") as fd:
        prompt = fd.read()

    # Configure the dataset
    data_files = {"train": args.train_data_path, "test": args.dev_data_path}
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.map(
        preprocess_function,
        fn_kwargs={"prompt": prompt},
        remove_columns=dataset["train"].column_names,
    )

    # Configure model and tokenizer
    if args.use_8bit or args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.use_8bit, load_in_4bit=args.use_4bit
        )
        dtype = None
    else:
        quantization_config = None
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.odel_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    # Setup chat template

    # Configure LoRA training
    peft_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Configure training args
    training_args = SFTConfig(
        # STF related args
        model_init_kwargs={"dtype": torch.bfloat16},
        packing=True,
        completion_only_loss=True,
        pad_to_multiple_of=8,
        use_liger_kernel=True,
        activation_offloading=True,
        max_length=2048,

        # General train args
        output_dir=args.save_path,
        overwrite_output_dir=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        weight_decay=0.1,
        max_steps=args.steps,
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=5,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        bf16=True,
        tf32=True,
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        torch_compile=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
