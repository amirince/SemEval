from langdetect import detect
import pandas as pd
import langid
from textblob import TextBlob
import csv
import string
import re


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


import pandas as pd
import string
import re
import csv


def clean_word(word):
    """
    Cleans a word by removing punctuation, converting to lowercase,
    filtering unwanted characters, and excluding numbers.
    """
    # Remove punctuation and convert to lowercase
    cleaned_word = word.translate(str.maketrans("", "", string.punctuation)).lower()

    # Remove unwanted characters using regex
    cleaned_word = re.sub(
        r"[^\w\s\u0400-\u04FF\u2C80-\u2CFF\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF\u2B50\u00A9\u00AE]",
        "",
        cleaned_word,
    )

    # Exclude numbers
    if cleaned_word.isdigit():
        return None  # Return None for numbers

    return cleaned_word


def get_bag_of_word(target_language, data_path):
    """
    Generates a bag of words from the dataset and saves them to a CSV file.
    """
    # Load the dataset and shuffle
    data_frame = pd.read_csv(data_path)
    sampled_df = data_frame.sample(n=100, random_state=42)

    # Extract unique cleaned words, excluding numbers
    temp_set = {
        clean_word(word)
        for sentence in sampled_df["text"]
        for word in sentence.split()
        if clean_word(word)  # Exclude None values
    }

    # Remove empty strings and sort the words
    data = sorted(temp_set)

    # Save the words to a CSV file with UTF-8 encoding
    output_path = f"lang_detection_lib/data/{target_language}.csv"
    with open(output_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["words"])
        writer.writerows([[word] for word in data])


# Process multiple datasets
for dataset in dataset_list:
    DATA_PATH = f"lang_id_data/{dataset}.csv"
    get_bag_of_word(dataset, DATA_PATH)
