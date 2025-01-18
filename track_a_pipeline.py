from example_analysis_pipeline import LangEvalAlgo
import asyncio
import pandas as pd
import time

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
    "afr.csv",
    "amh.csv",
    "deu.csv",
    "eng.csv",
    "oro.csv",
    "ptbr.csv",
    "rus.csv",
    "som.csv",
    "tir.csv",
]


for dataset in dataset_list:
    data = pd.read_csv(f"public_data/train/track_a/{dataset}")
    data = data.head(1)
    new_df = []
    for index, row in data.iterrows():
        example = row["text"]
        print(row)
        resp = asyncio.run(evaluator.run(example=example))

        response_one_hot = [0, 0, 0, 0, 0]
        for index, emotion in enumerate(poss_emo):
            if emotion in resp:
                row[emotion] = 1
            else:
                row[emotion] = 0

        print(row)
        new_df.append(row.to_dict())
    new_data = pd.DataFrame(new_df)
    new_data.to_csv(f"results_complete/{dataset}.csv", index=False)
    print("written to folder")

    time.sleep(300)
