import argparse
import json
import os
import tempfile

import openai
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-path", type=str, required=True, help="Path to the prompt file to use."
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the QG csv file to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        choices=["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07"],
        help="Path to the prompt file to use.",
    )

    args = parser.parse_args()
    return args


def create_batch_file(df: pl.DataFrame, prompt: str, model: str, client):
    requests: list[dict] = []
    for index, row in enumerate(df.iter_rows(named=True)):
        request = {
            "custom_id": f"row-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt.format(
                            question=row["question"], context=row["context"]
                        ),
                    },
                ],
            },
        }
        requests.append(request)  # store as dict, not string

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as tmp_file:
        for record in requests:
            tmp_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )  # now dumps once
        temp_path = tmp_file.name

    batch_input_file = client.files.create(file=open(temp_path, "rb"), purpose="batch")
    return batch_input_file


def main() -> None:
    args = parse_args()

    with open(args.prompt_path, "r") as fd:
        prompt = fd.read()

    df = pl.read_csv(args.data_path)

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG"),
        project=os.environ.get("OPENAI_PROJECT"),
    )

    batch_input_file = create_batch_file(df, prompt, args.model, client)
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )
    print(batch_job)


if __name__ == "__main__":
    main()
