from example_analysis_pipeline import LangEvalAlgo
import asyncio
import pandas as pd
import time
import os

### For testing purposes:
judge_model = "llama3.1:8b"  # this is the main judge model.

# jurors = (
#     "llama3.2:1b",
#     "llama3.2:3b",
#     "llama2:latest",
#     "llama3:latest",
# )  # need to cut this down to one.

juror_model = "llama3.2:3b"

poss_emo = [
    "Anger",
    "Fear",
    "Joy",
    "Sadness",
    "Surprise",
]

dataset_list = [
    "afr",
    "amh",
    "arq",
    "ary",
    "chn",
    "deu",
    "eng",
    "esp",
    "hau",
    "hin",
    "ibo",
    "ind",
    "jav",
    "kin",
    "mar",
    "orm",
    "pcm",
    "ptbr",
    "ptmz",
    "ron",
    "rus",
    "som",
    "sun",
    "swa",
    "swe",
    "tat",
    "tir",
    "ukr",
    "vmw",
    "xho",
    "yor",
    "zul",
]


def save_partial_data(buffer, dataset_name, part_number):
    """Saves buffered rows to a CSV file."""
    if buffer:
        partial_data = pd.DataFrame(buffer)
        output_file = f"results_complete/{dataset_name}_part{part_number}.csv"
        partial_data.to_csv(output_file, index=False)
        print(f"Saved {len(buffer)} rows to {output_file}")
        return True
    return False


evaluator = LangEvalAlgo(
    juror_model=juror_model, judge_model=judge_model, possible_emotions=poss_emo
)

for dataset in dataset_list:
    file_path = f"public_data_test/track_a/test/{dataset}.csv"

    if not os.path.exists(file_path):
        print("path not found")
        continue

    data = pd.read_csv(file_path)

    buffer = []
    batch_size = 50

    output_file = f"results_complete/{dataset}.csv"

    for index, row in data.iterrows():
        print(f"Processing Example: {index}")
        example = row["text"]
        resp = asyncio.run(evaluator.run_jurors(example=example))

        row["lanaguage"] = resp[0]
        row["juror_1"] = resp[1]
        row["juror_2"] = resp[2]
        row["juror_3"] = resp[3]
        row["juror_4"] = resp[4]

        buffer.append(row.to_dict())

        # Save every 50 rows
        if len(buffer) >= batch_size:
            pd.DataFrame(buffer).to_csv(
                output_file,
                mode="a",
                index=False,
                header=not os.path.exists(output_file),
            )
            print(f"Appended {len(buffer)} rows to {output_file}")
            buffer = []  # Reset buffer

    # Save any remaining rows
    if buffer:
        pd.DataFrame(buffer).to_csv(
            output_file, mode="a", index=False, header=not os.path.exists(output_file)
        )
        print(f"Appended remaining {len(buffer)} rows to {output_file}")

    print(f"Completed processing for dataset: {dataset}")

    # for dataset in dataset_list:
    # data = pd.read_csv(f"public_data/train/track_a/{dataset}")
    # data = data.head(2)
    # buffer = []  # Collect processed rows
    # batch_size = 50
    # output_file = f"results_complete/{dataset}"  # Single output CSV for this dataset

    # for index, row in data.iterrows():
    #     print(f"Processing Example: {index}")
    #     example = row["text"]
    #     resp = asyncio.run(evaluator.run_jurors(example=example))

    #     # Update row with emotion results
    #     for emotion in poss_emo:
    #         row[emotion] = 1 if emotion in resp else 0

    #     buffer.append(row.to_dict())

    #     # Save every 50 rows
    #     if len(buffer) >= batch_size:
    #         pd.DataFrame(buffer).to_csv(
    #             output_file,
    #             mode="a",
    #             index=False,
    #             header=not os.path.exists(output_file),
    #         )
    #         print(f"Appended {len(buffer)} rows to {output_file}")
    #         buffer = []  # Reset buffer

    # # Save any remaining rows
    # if buffer:
    #     pd.DataFrame(buffer).to_csv(
    #         output_file, mode="a", index=False, header=not os.path.exists(output_file)
    #     )
    #     print(f"Appended remaining {len(buffer)} rows to {output_file}")

    # print(f"Completed processing for dataset: {dataset}")
    # time.sleep(150)
