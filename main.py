"""Run Jurors then Judges to produce final results."""

import asyncio
import time
import os
import pandas as pd
from example_analysis_pipeline import LangEvalAlgo
from config import (
    JUDGE_MODEL,
    JUROR_MODEL,
    DATASET_LIST,
    BATCH_SIZE,
    POSSIBLE_EMOTIONS,
    DATA_PATH,
)

# Initializing Evaluator
evaluator = LangEvalAlgo(
    juror_model=JUROR_MODEL,
    judge_model=JUDGE_MODEL,
    possible_emotions=POSSIBLE_EMOTIONS,
)


def juror_analysis():
    """Run juror analysis for all examples in all
    datasets and store them to intermediate location"""

    for dataset in DATASET_LIST:
        file_path = f"{DATA_PATH}/{dataset}.csv"

        if not os.path.exists(file_path):
            print("path not found")
            continue

        data = pd.read_csv(file_path)
        buffer = []

        # Outputs the juror results to intermediate folder
        output_file = f"intermediate_results/{dataset}.csv"

        for index, row in data.iterrows():
            print(f"Processing Example: {index}")
            example = row["text"]

            start = time.time()
            resp = asyncio.run(evaluator.run_jurors(example=example))
            end = time.time()

            print(end - start)

            row["language"] = resp[0]
            row["juror_1"] = resp[1]
            row["juror_2"] = resp[2]
            row["juror_3"] = resp[3]
            row["juror_4"] = resp[4]

            buffer.append(row.to_dict())

            # Save every 50 rows
            if len(buffer) >= BATCH_SIZE:
                pd.DataFrame(buffer).to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(output_file),
                )
                buffer = []  # Flush Buffer

        # Save any remaining rows
        if buffer:
            pd.DataFrame(buffer).to_csv(
                output_file,
                mode="a",
                index=False,
                header=not os.path.exists(output_file),
            )

        print(f"Completed processing for dataset: {dataset}")


def judge_analysis():
    """Run judge analysis for all examples in all datasets
    and store them to intermediate location"""

    for dataset in DATASET_LIST:
        file_path = f"intermediate_results/{dataset}.csv"

        if not os.path.exists(file_path):
            print("path not found")
            continue

        data = pd.read_csv(file_path)
        buffer = []

        # Outputs the juror results to intermediate folder
        output_file = f"final_results/{dataset}.csv"

        for index, row in data.iterrows():
            print(f"Processing Example: {index}")
            example = row["text"]
            example_lang = row["language"]

            # Get juror assessments from intermediate file
            juror_assesments = []
            juror_assesments.append(row["juror_1"])
            juror_assesments.append(row["juror_2"])
            juror_assesments.append(row["juror_3"])
            juror_assesments.append(row["juror_4"])

            judge_response = asyncio.run(
                evaluator.run_judge(
                    example=example,
                    language=example_lang,
                    juror_assessments=juror_assesments,
                )
            )

            judge_response = judge_response.lower()

            row.drop("juror_1")
            row.drop("juror_2")
            row.drop("juror_3")
            row.drop("juror_4")

            for emotion in POSSIBLE_EMOTIONS:
                row[emotion] = int(emotion.lower() in judge_response)

            buffer.append(row.to_dict())

            # Save every 50 rows
            if len(buffer) >= BATCH_SIZE:
                pd.DataFrame(buffer).to_csv(
                    output_file,
                    mode="a",
                    index=False,
                    header=not os.path.exists(output_file),
                )
                buffer = []  # Flush Buffer

        # Save any remaining rows
        if buffer:
            pd.DataFrame(buffer).to_csv(
                output_file,
                mode="a",
                index=False,
                header=not os.path.exists(output_file),
            )

        print(f"Completed processing for dataset: {dataset}")


juror_analysis()
judge_analysis()
