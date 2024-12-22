import pandas as pd
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
    "tir": "ti",
}


class TextClassify:
    def __init__(self) -> None:
        # Load language mappings
        self._load_mapping()

    def _load_mapping(self):
        # Load the word sets for each language from CSV files
        for entry in dataset_list:
            df = pd.read_csv(f"lang_detection_lib/data/{entry}.csv")
            dataset_list[entry] = set(df["words"])

    def classify(self, text):
        # Split and preprocess text into words
        words = text.split()
        temp_set = set()

        for word in words:
            cleaned_word = word.translate(str.maketrans("", "", string.punctuation))
            cleaned_word = cleaned_word.lower()
            cleaned_word = re.sub(
                r"[^\w\s\u0400-\u04FF\u2C80-\u2CFF\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF\u2B50\u00A9\u00AE]",
                "",
                cleaned_word,
            )
            temp_set.add(cleaned_word)

        # print("Cleaned words:", temp_set)  # Debug: Check preprocessing

        language_match_percentages = {}

        for language, word_set in dataset_list.items():
            match_count = 0
            matches = []

            for word in temp_set:
                if word in word_set:
                    match_count += 1
                    matches.append(word)  # Track matched words for debugging

            # Calculate the match percentage
            match_percentage = (
                (match_count / len(temp_set)) * 100 if len(temp_set) > 0 else 0
            )
            language_match_percentages[language] = match_percentage

            # print(f"Matches for {language}: {matches}")  # Debug: Check matches

        most_likely_language = max(
            language_match_percentages, key=language_match_percentages.get
        )
        # print("Language match percentages:", language_match_percentages)  # Debug: Check percentages
        return most_likely_language

    # def classify(self, text):

    #     # Split the text into words
    #     words = text.split()
    #     temp_set = set()

    #     # Pre-process each word (removing punctuation, converting to lowercase, etc.)
    #     for word in words:
    #         cleaned_word = word.translate(str.maketrans("", "", string.punctuation))
    #         cleaned_word = cleaned_word.lower()
    #         cleaned_word = re.sub(r'[^\w\s\u0400-\u04FF\u2C80-\u2CFF\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2600-\u26FF\u2700-\u27BF\u2300-\u23FF\u2B50\u00A9\u00AE]', '', cleaned_word)
    #         temp_set.add(cleaned_word)

    #     language_match_percentages = {}

    #     # Check each language's word set for matches
    #     for language, word_set in dataset_list.items():
    #         match_count = 0

    #         # Compare each cleaned word in the sentence to the language's word set
    #         for word in temp_set:
    #             if word in word_set:
    #                 match_count += 1

    #         # Calculate the match percentage for this language
    #         match_percentage = (match_count / len(temp_set)) * 100 if len(temp_set) > 0 else 0
    #         language_match_percentages[language] = match_percentage

    #     most_likely_language = max(language_match_percentages, key=language_match_percentages.get)
    #     return (most_likely_language)
