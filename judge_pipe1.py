from example_analysis_pipeline import LangEvalAlgo
import asyncio
import pandas as pd
import time
import os
from ollama import AsyncClient
from ollama import Client


JUDGE_TEMPLATE = """

## Task: Final Emotion Determination

**Review the Juror Assessments:**

Carefully review the emotion assessments provided by the Jurors. 
Pay attention to the range of emotions identified, the frequency of specific emotions, and the level of confidence expressed by each Juror.

**Consider the Following:**

1. **Consensus:**
   * Identify emotions that have been consistently selected by multiple Jurors.
   * Prioritize emotions with strong consensus.

2. **Confidence Levels:**
   * Assess the confidence levels expressed by the Jurors.
   * Give more weight to emotions that have been identified with high confidence.

3. **Nuance and Complexity:**
   * Consider the possibility of multiple emotions or complex emotional states.
   * Look for subtle cues and underlying feelings that may not be explicitly stated.

** For context:**
1. This is the sample text the jurors were asked to classify: "{{text}}".
2. The languge of the above text is {{lang_id}}.
3. The possible emotions invoked by the above text are: {{possible_emotions}}.

**Juror Assesments:**

{{juror_assessment}}

**Make a Final Decision:**

Based on your analysis, determine the primary emotion(s) conveyed in the text. 

Please only provide the final emotion(s) in your response. Do not to explain your thought process.
"""


judge_model = "deepseek-r1:32b"

poss_emo = ["Anger", "Fear", "Joy", "Sadness", "Surprise", "Disgust"]

dataset_list = [
    # "afr",
    # "amh",
    # "arq",
    # "ary",
    "chn",
    "ary",
    # "deu",
    # "eng",model='llama3.2:3b'
    # "esp",
    # "hau",
    # "hin",
    # "ibo",
    # "kin",
    # "mar",
    # "orm",
    # "pcm",
    # "ptbr",
    # "ptmz",
    # "ron",
    # "rus",
    # "som",
    # "sun",
    # "swa",
    # "swe",
    # "tat",
    # "tir",
    # "ukr",
    # "vmw",
    # "xho",
    # "yor",
    # "zul",
]


def create_judge_template(example, juror_assessments, example_lang):

    formatted_assessments = "\n".join(
        f"Juror{value}\nResponse: {str(emotions)}\n"
        for value, emotions in enumerate(juror_assessments)
    )

    judge_template = (
        JUDGE_TEMPLATE.replace("{{juror_assessment}}", formatted_assessments)
        .replace("{{lang_id}}", example_lang)
        .replace("{{possible_emotions}}", str(poss_emo))
        .replace("{{text}}", example)
    )
    return judge_template


async def run_judge(example: str, juror_assessments: list, example_lang: str):

    # Create the judge prompt make final decision
    prompt = create_judge_template(
        example=example, juror_assessments=juror_assessments, example_lang=example_lang
    )
    judge_response = await query_model(judge_model, prompt)
    return judge_response


async def query_model(model, prompt):
    response = await AsyncClient().chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response["message"]["content"]


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
    file_path = f"results_complete/{dataset}.csv"

    if not os.path.exists(file_path):
        print("path not found")
        continue

    data = pd.read_csv(file_path)

    buffer = []
    batch_size = 25

    output_file = f"judge_complete/{dataset}.csv"

    for index, row in data.iterrows():
        print(f"Processing Example: {index}")
        example = row["text"]
        example_lang = row["lanaguage"]

        juror_assesments = []
        juror_assesments.append(row["juror_1"])
        juror_assesments.append(row["juror_2"])
        juror_assesments.append(row["juror_3"])
        juror_assesments.append(row["juror_4"])

        start = time.time()
        resp = asyncio.run(
            run_judge(
                example=example,
                juror_assessments=juror_assesments,
                example_lang=example_lang,
            )
        )
        end = time.time()

        resp = resp.lower()

        row["juror_1"] = " "
        row["juror_2"] = " "
        row["juror_3"] = " "
        row["juror_4"] = " "

        if "disgust" in resp:
            row["disgust"] = 1
        else:
            row["disgust"] = 0

        if "anger" in resp:
            row["anger"] = 1
        else:
            row["anger"] = 0

        if "fear" in resp:
            row["fear"] = 1
        else:
            row["fear"] = 0

        if "joy" in resp:
            row["joy"] = 1
        else:
            row["joy"] = 0

        if "sadness" in resp:
            row["sadness"] = 1
        else:
            row["sadness"] = 0

        if "surprise" in resp:
            row["surprise"] = 1
        else:
            row["surprise"] = 0

        buffer.append(row.to_dict())
        print("Time:", end - start)
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
