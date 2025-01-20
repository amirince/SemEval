from example_analysis_pipeline import LangEvalAlgo
import asyncio
import pandas as pd
import time
import os

### For testing purposes:
judge_model = "llama3.1:8b"

jurors = ["llama3.2:1b", "llama3.2:3b", "llama2:latest", "llama3:latest"]

poss_emo = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]


evaluator = LangEvalAlgo(
    juror_models=jurors, judge_model=judge_model, possible_emotions=poss_emo
)

# resp = asyncio.run(evaluator.run(example=example))


# for emotion in poss_emo:
#     if emotion in resp:
#         print(1)
#     else:
#         print(0)


dataset_list = [
    # "afr.csv",
    "amh.csv",
    "deu.csv",
    "eng.csv",
    "oro.csv",
    "ptbr.csv",
    "rus.csv",
    "som.csv",
    "tir.csv",
]


# for dataset in dataset_list:
#     data = pd.read_csv(f"public_data/train/track_a/{dataset}")
#     new_df = []
#     for index, row in data.iterrows():
#         print("Example: " , index)
#         example = row["text"]
#         resp = asyncio.run(evaluator.run(example=example))

#         response_one_hot = [0, 0, 0, 0, 0]
#         for index, emotion in enumerate(poss_emo):
#             if emotion in resp:
#                 row[emotion] = 1
#             else:
#                 row[emotion] = 0

#         new_df.append(row.to_dict())
#     new_data = pd.DataFrame(new_df)
#     new_data.to_csv(f"results_complete/{dataset}.csv", index=False)
#     print("written to folder")

#     time.sleep(300)


def save_partial_data(buffer, dataset_name, part_number):
    """Saves buffered rows to a CSV file."""
    if buffer:
        partial_data = pd.DataFrame(buffer)
        output_file = f"results_complete/{dataset_name}_part{part_number}.csv"
        partial_data.to_csv(output_file, index=False)
        print(f"Saved {len(buffer)} rows to {output_file}")
        return True
    return False

for dataset in dataset_list:
    data = pd.read_csv(f"public_data/train/track_a/{dataset}")
    buffer = []  # Collect processed rows
    batch_size = 50
    output_file = f"results_complete/{dataset}"  # Single output CSV for this dataset

    for index, row in data.iterrows():
        print(f"Processing Example: {index}")
        example = row["text"]
        resp = asyncio.run(evaluator.run(example=example))

        # Update row with emotion results
        for emotion in poss_emo:
            row[emotion] = 1 if emotion in resp else 0

        buffer.append(row.to_dict())

        # Save every 50 rows
        if len(buffer) >= batch_size:
            pd.DataFrame(buffer).to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            print(f"Appended {len(buffer)} rows to {output_file}")
            buffer = []  # Reset buffer

    # Save any remaining rows
    if buffer:
        pd.DataFrame(buffer).to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        print(f"Appended remaining {len(buffer)} rows to {output_file}")

    print(f"Completed processing for dataset: {dataset}")
    time.sleep(150)
