import argparse
import os
import re
import tomllib
from typing import NamedTuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FewShotExample(NamedTuple):
    context: str
    question: str
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the QA CSV to evaluate."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./answers.csv",
        help="Path to save the results",
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
        "--use-vllm",
        action="store_true",
        help="Wheter to use vllm for model inference.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["bitsandbytes", "modelopt", "fp8"],
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
    parser.add_argument(
        "--num-of-shots",
        type=int,
        default=0,
        help="Num of shots to use in a few-shot prompt.",
    )

    args = parser.parse_args()

    if args.quantization is not None and not args.use_vllm:
        raise parser.error("--use-vllm is required when using --quantization")

    return args


def load_model(
    model_name: str,
    use_8bit: bool = False,
    use_4bit: bool = False,
    device: str = "cuda",
):
    if use_8bit or use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit, load_in_4bit=use_4bit
        )
        dtype = None
    else:
        quantization_config = None
        dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=dtype,
        attn_implementation="eager",
        device_map=device,
    )
    model = torch.compile(
        model, backend="inductor", mode="reduce-overhead", fullgraph=True
    )

    return model, tokenizer


def build_few_shot_example(shot_list: list[FewShotExample]) -> str:
    few_shot_prompt = """
    Example {i}:

    Example Passage: {context}

    Example Sentence: {sentence}

    Example Response: {response}
    """
    prompts: list[str] = []
    for i, shot in enumerate(shot_list, 1):
        example_prompt = few_shot_prompt.format(
            i=i, context=shot.context, sentence=shot.question, response=shot.answer
        )
        prompts.append(example_prompt)
    return "\n\n".join(prompts)


def evaluate_question(
    model,
    tokenizer,
    df_slice: pl.DataFrame,
    prompt: str,
    few_shots: list[FewShotExample],
    use_vllm: bool = False,
) -> str | list[str]:
    few_shot_prefix = build_few_shot_example(few_shots)
    messages = [
        [
            {
                "role": "user",
                "content": prompt.format(
                    question=row["question"],
                    context=row["context"],
                    few_shot=few_shot_prefix,
                ),
            }
        ]
        for row in df_slice.iter_rows(named=True)
    ]
    encoding = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=not use_vllm,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=not use_vllm,
    )

    if use_vllm:
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        outputs = model.generate(encoding)
        return [output.outputs[0].text for output in outputs]
    else:
        prompt_length = encoding["input_ids"].size(1)
        outputs = model.generate(
            **encoding.to(model.device),
            do_sample=True,
            temperature=1.0,
            max_new_tokens=512,
            cache_implementation="static",
            renormalize_logits=True,
        )
        # pre-process inputs
        return tokenizer.batch_decode(
            outputs[:, prompt_length:], skip_special_tokens=True
        )


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
    # if "Answer:" in output:
    #     answer_start_index = output.find("Answer: ")
    #     # Honestly the part below with re is kinda useless but I like it
    #     answer_start = output[answer_start_index:]  # We slice to only get the answer
    if "Not answerable" in output:
        success = 1
        answer = None
    else:
        match = re.search(r"(?<=<ans>).*(?=<\/ans>)", output)
        if match is not None:
            success = 0
            answer = match.group().strip()
        else:
            success = -1
            answer = None

    return success, answer


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

    # sum of y_true would give the count of non-answerable questions
    # and impossible_pred contains all the predictions for those non-answerable questions
    # so summing impossible_pred gives us how many question the model decided that are non-answerable from the non-answerable set
    impossible_pred = [
        pred for pred, label in zip(y_pred, y_true) if label == 1 and pred != 2
    ]
    tpr = sum(impossible_pred) / sum(y_true)

    # Maybe in the future we'll also calculate stuff like lexical metrics
    # Such as ROUGE, BLEU, and so forth.

    return {"accuracy": acc, "f1": f1, "tpr": tpr, "fail_rate": fail_rate}


def main() -> None:
    args = parse_args()

    with open(args.config_path, "rb") as fd:
        config = tomllib.load(fd)

    with open(config["prompt_path"], "r") as fd:
        prompt = fd.read()

    # load the model + tokenizer
    if args.use_vllm:
        model = LLM(
            model=config["model_name"],
            enable_prefix_caching=False,
            max_model_len=2048,
            quantization=args.quantization,
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    else:
        model, tokenizer = load_model(
            config["model_name"], args.use_8bit, args.use_4bit, args.device
        )

    df = pl.read_csv(args.data_path).with_row_index()

    llm_answers: list[str] = []
    labels: list[tuple[int, str | None]] = []
    for i, df_slice in enumerate(tqdm(df.iter_slices(n_rows=args.batch_size)), 1):
        few_shots: list[FewShotExample] = []
        if args.num_of_shots > 0:
            rest_df = df.join(df_slice, on="index", how="anti")
            sampled_rows = rest_df.sample(n=args.num_of_shots, with_replacement=False)
            for sample_row in sampled_rows.iter_rows(named=True):
                example = FewShotExample(
                    context=sample_row["context"],
                    question=sample_row["question"],
                    answer=sample_row["answer"]
                    if sample_row["is_impossible"] == 0
                    else "Not answerable",
                )
                few_shots.append(example)
        answers = evaluate_question(
            model,
            tokenizer,
            df_slice,
            prompt=prompt,
            few_shots=few_shots,
            use_vllm=args.use_vllm,
        )
        for ii, row in enumerate(df_slice.iter_rows(named=True)):
            labels.append((row["is_impossible"], row["answer"]))
            llm_answers.append(answers[ii])
        # if i == 10:
        #     break

    llm_binary_preds = [convert_llm_output_to_binary(answer) for answer in llm_answers]
    metrics = compute_metrics(llm_binary_preds, labels)
    print(metrics)

    df = df.with_columns(pl.Series(name="llm_replies", values=llm_answers))
    df.write_csv(args.save_path)
    # Next steps:
    # 4. Convert the script to sbatch format to run.

    # 5. Maybe convert the sbatch to array format.

    # 6. Test multiple SLMs.

    # 7. Extract the code to mutiple files, utils, data handlers.

    # 9. Imporve the prompt.


if __name__ == "__main__":
    main()
