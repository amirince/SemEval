from langdetect import detect
import pandas as pd
import langid
from textblob import TextBlob
import csv
import string
import re


dataset_list = {
    "afr": "af",
    "amh": "am",
    "deu": "de",
    "eng": "en",
    "ind": "in",
    "jav": "jav",
    "oro": "om",
    "ptbr": "pt",
    "rus": "ru",
    "som": "so",
    "sun": "su",
    "tir": "ti"
}


def get_bag_of_word(target_language, data_path):
    data_frame = pd.read_csv(data_path)
    data_frame = data_frame.head(100)

    temp_set = set()
    for sentence in data_frame["text"]:
        words = sentence.split()
        
        for word in words:

            #pre-process words
            cleaned_word = word.translate(str.maketrans("", "", string.punctuation))
            cleaned_word = cleaned_word.lower()
            cleaned_word = re.sub(r'[^\w\s\u0400-\u04FF\u2C80-\u2CFF\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF\u2B50\u00A9\u00AE]', '', cleaned_word)
            
            temp_set.add(cleaned_word)

    data = list(temp_set)
    data = data[1::]
    with open(f"lang_detection_lib/data/{target_language}.csv", "w") as file:
    
        writer = csv.writer(file)
        writer.writerow(["words"])

        # Write each element of the set as a new row
        for item in data:
            writer.writerow([item])

for dataset, target_language in dataset_list.items():
    DATA_PATH = f"public_data/dev/track_c/{dataset}_c.csv"
    get_bag_of_word(dataset,DATA_PATH)
    