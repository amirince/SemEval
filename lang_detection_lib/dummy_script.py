import os

# Specify the directory containing the .csv files
directory_path = "lang_id_data"  # Replace with your directory path

# List all files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith(".csv"):
        # Remove the '.csv' extension and print the name
        print(f"'{file_name[:-4]}',")
