import pandas as pd


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

def print_metrics(PATH, LABEL):

    # Open the file
    try:
        df = pd.read_csv(PATH, sep=",")
    except FileNotFoundError:
        print("File Not Found!")
        return
    
    #Calc metrics
    total_samples = len(df)
    count = df["prediction"].str.lower().eq(LABEL).sum()


    print(f"{LABEL}: {count}/{total_samples}")

models = ['llama2_7b', "llama2_latest", "llama3", "llama3.1_8b", "llama3.2_1b", "llama3.2_3b"]



for model in models:
    print(f"\nModel: {model}")
    for lang, label in dataset_list.items():
        DataPath = f"results/track_a/{model}/{lang}.csv"
        print_metrics(DataPath, label)
