import argparse
import os
import time

import polars as pl
from google import genai
from tqdm.auto import tqdm


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
        default="gemini-2.5-flash",
        choices=["gemini-2.5", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        help="Path to the prompt file to use.",
    )

    args = parser.parse_args()
    return args


def send_single_response(
    client, prompt: str, context: str, question: str, model: str = "gemini-2.5-flash"
) -> str:
    response = client.models.generate_content(
        model=model, contents=prompt.format(question=question, context=context)
    )
    answer = response.text

    return answer


def main() -> None:
    args = parse_args()

    with open(args.prompt_path, "r") as fd:
        prompt = fd.read()

    df = pl.read_csv(args.data_path)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    answers: list[str] = []
    for row in tqdm(df.iter_rows(named=True)):
        answer = send_single_response(
            client, prompt, row["context"], row["question"], args.model
        )
        answers.append(answer)
        time.sleep(0.2)

    df = df.with_columns(pl.Series(name="gemini_replies", values=answers))
    df.write_csv("answers.csv")


if __name__ == "__main__":
    main()
