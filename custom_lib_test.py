from langdetect import detect
import pandas as pd
import langid
from textblob import TextBlob
import csv
import string
import re
from lang_detection_lib.lang_classify import TextClassify


dataset_list = {
    "afr": "af",
    "amh": "am",
    "deu": "de",
    "eng": "en",
    # "ind": "in",
    # "jav": "jav",
    "oro": "om",
    "ptbr": "pt",
    "rus": "ru",
    "som": "so",
    "sun": "su",
    "tir": "ti",
}

classifier = TextClassify()
for dataset, target_language in dataset_list.items():
    DATA_PATH = f"public_data/train/track_a/{dataset}.csv"
    df = pd.read_csv(DATA_PATH)
    length = len(df)
    count = 0
    for entry in df["text"]:

        label = classifier.classify(entry)

        if label == dataset:
            count += 1
            continue

        try:
            label = detect(entry)

            if label == dataset:
                count += 1
                continue

        except:
            pass

        try:
            label = detect(entry)
            # print(label)
            if label == dataset:
                count += 1
                continue

        except:
            pass

    print(f"Dataset {dataset}")
    print(f"{count}/{length}")
    print(count / length)
    print("")
