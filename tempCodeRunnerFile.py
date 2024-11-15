import pandas as pd
import asyncio
import os
import ollama
from ollama import AsyncClient
import time
import re

DATA_PATH = "data_sets/ary/test.tsv"

MODEL = "llama3.1:8b"

PROMPT = """You are a highly skilled language classification agent. Your task is to determine what the laguage is of the given text. You response should be in the format:

{language}
"""

EXAMPLE_COL = "text"
LABEL_COL = "sentiment"


class RunInference:

    def __init__(
        self,
        data_path,
        dataset_name,
        batch_size=50,
        model="llama3.2:1b",
        prompt=PROMPT,
        example_col=EXAMPLE_COL,
        label_col=LABEL_COL,
        target_language="",
    ) -> None:
        self.model = model
        self.model_name = re.sub(r'[<>:"/\\|?*]', "_", model)
        self.prompt = prompt
        self.label = label_col
        self.example_col = example_col
        self.dataset_name = dataset_name
        self.data = pd.read_csv(data_path, delimiter=",")
        self.data_path = data_path
        self.batch_size = batch_size
        self.all_labels = []
        self.target_lang = target_language

        self.data = self.data.head(10)

    async def classify_text(self, example):
        response = await AsyncClient().chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt,
                },
                {
                    "role": "user",
                    "content": example,
                },
            ],
        )

        # Parsing sentiment from LLM response:
        print("*********************")
        print(response)
        sentiment = response["message"]["content"]

        sentiment = sentiment.strip()
        sentiment = sentiment.lower()

        if self.target_lang.lower() in sentiment:
            return self.target_lang.lower()
        else:
            return "NIL"

    def _save_results(self, current_batch_df):
        if not os.path.exists(f"LangAnalysis/results/track_a/{self.model_name}"):
            os.makedirs(f"LangAnalysis/results/track_a/{self.model_name}")

        output_path = (
            f"LangAnalysis/results/track_a/{self.model_name}/{self.dataset_name}.csv"
        )

        results_df = pd.DataFrame(self.all_labels, columns=["prediction"])
        current_batch_df["prediction"] = results_df["prediction"][
            -len(current_batch_df) :
        ]
        current_batch_df.to_csv(
            output_path,
            sep=",",
            index=False,
            mode="a",
            header=not os.path.exists(output_path),
        )

        print(f"Batch results saved to {output_path}")

    async def process_batch(self, batch_df):
        max_retries = 1
        labels = []

        for idx, row in batch_df.iterrows():
            retries = 0
            success = False
            label = None

            while not success and retries < max_retries:
                try:
                    print("Example: ", idx)
                    label = await self.classify_text(row[self.example_col])
                    success = True
                except Exception as e:
                    retries += 1
                    print(f"Error classifying example {idx}")

            if success:
                labels.append(label)
            else:
                labels.append("NIL")
                print(f"Failed to classify example {idx} after {max_retries} attempts.")

        self.all_labels.extend(labels)
        self._save_results(batch_df)

    async def run(self):

        start_time = time.time()

        total_rows = len(self.data)
        for start in range(0, total_rows, self.batch_size):
            end = min(start + self.batch_size, total_rows)
            batch_df = self.data.iloc[start:end]
            await self.process_batch(batch_df)

        end_time = time.time()
        duration = end_time - start_time
        print(f"Processing completed in {duration:.2f} seconds.")


dataset_list = {
    "afr": "afrikaans",
    "amh": "amharic",
    "deu": "german",
    "eng": "english",
    # "ind": "indonesian",
    # "jav": "javanese",
    "oro": "oromo",
    "ptbr": "portugese",
    "rus": "russian",
    "som": "somali",
    "sun": "sundanese",
    "tir": "tigrinya",
}

for dataset, target_language in dataset_list.items():
    DATA_PATH = f"LangAnalysis\public_data/dev/track_a/{dataset}_a.csv"
    classifier = RunInference(
        data_path=DATA_PATH,
        dataset_name=dataset,
        batch_size=1,
        model="llama3.2:1b",
        prompt=PROMPT,
        target_language=target_language,
    )

    asyncio.run(classifier.run())
