import argparse
import os
import re
import tomllib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the QA CSV to evaluate."
    )
    parser.add_argument(
        "--use-8bit",
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
        "--batch-size",
        type=int,
        default=1,
        help="Batch size to use for inference. Default is 1.",
    )

    args = parser.parse_args()

    return args


def load_model(model_name: str, use_8bit: bool = False, device: str = "cuda"):
    if use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None
    else:
        quantization_config = None
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map=device,
    )
    model = torch.compile(
        model, backend="inductor", mode="reduce-overhead", fullgraph=True
    )
    return model, tokenizer


def evaluate_question(model, tokenizer, df_slice: pl.DataFrame, prompt: str) -> str:
    messages = [
        [
            {
                "role": "user",
                "content": prompt.format(
                    question=row["question"], context=row["context"]
                ),
            }
        ]
        for row in df_slice.iter_rows(named=True)
    ]
    encoding = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_length = encoding["input_ids"].size(1)

    outputs = model.generate(
        **encoding,
        do_sample=True,
        temperature=1.0,
        max_new_tokens=1024,
    )
    # pre-process inputs
    return tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)


def convert_llm_output_to_binary(output: str) -> tuple[int, str | None]:
    """
    This method suppose to convert the LLM output to 0 or 1
    Depending on the prompt and LLM output
    Overall the default is that if `Answer:` prefix exists
    and there's an answer wrapped in <ans> tokens
    And the question is answerable and thus good
    Otherwise if "not answerable" exists, the LLM thinks the question is bad
    Any other case we count it as LLM failure to comply
    We return 1 if the question is not answerable, 0 otherwise.
    """
    if "Answer:" in output:
        answer_start_index = output.find("Answer: ")
        # Honestly the part below with re is kinda useless but I like it
        answer_start = output[answer_start_index:]  # We slice to only get the answer
        match = re.search(r"(?<=<ans>).*(?=<ans>)", answer_start)
        if match is not None:
            success = 1
            answer = match.group().strip()
        else:
            success = 1
            answer = None
    elif "Not Answerable" in output:
        success = 0
        answer = None
    else:
        success = -1
        answer = None

    return 1 - success, answer


def compute_metrics(
    preds: list[tuple[int, str | None]], labels: list[tuple[int, str | None]]
) -> dict[str, float]:
    y_pred = [score for score, text in preds]
    y_true = [score for score, text in labels]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true, y_pred, average="macro"
    )  # It's 'multi-class' cause of failure rates
    fail_rate = y_pred.count(-1) / len(
        y_true
    )  # Check how many times the LLM failed to comply

    # Maybe in the future we'll also calculate stuff like lexical metrics
    # Such as ROUGE, BLEU, and so forth.

    return {"accuracy": acc, "f1": f1, "fail_rate": fail_rate}


def main() -> None:
    args = parse_args()

    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    with open(config["prompt_path"], "r") as fd:
        prompt = fd.read()

    # load the model + tokenizer
    model, tokenizer = load_model(config["model_name"], args.use_8bit, args.device)

    df = pl.read_csv(args.data_path)

    llm_answers: list[str] = []
    labels: list[tuple[int, str | None]] = []
    for i, df_slice in enumerate(tqdm(df.iter_slices(n_rows=args.batch_size)), 1):
        # print("##########################################")
        # print(f"Question: {row['question']}")
        # print(f"Answer: {row['answer']}")
        # print(f"Is impossible: {row['is_impossible']}")
        answers = evaluate_question(model, tokenizer, df_slice, prompt=prompt)
        # print(f"{answer}")
        # print("##########################################")
        for i, row in enumerate(df_slice.iter_rows(named=True)):
            labels.append((row["is_impossible"], row["answer"]))
            llm_answers.append(answers[i])
        if i == 10:
            break

    print(f"{len(llm_answers)=}")
    print(f"{len(labels)=}")

    llm_binary_preds = [convert_llm_output_to_binary(answer) for answer in llm_answers]
    metrics = compute_metrics(llm_binary_preds, labels)
    print(metrics)

    # Next steps:
    # 4. Convert the script to sbatch format to run.

    # 5. Maybe convert the sbatch to array format.

    # 6. Test multiple SLMs.

    # 7. Extract the code to mutiple files, utils, data handlers.

    # 9. Imporve the prompt.


if __name__ == "__main__":
    main()
