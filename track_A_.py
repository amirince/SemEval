import os
import aiofiles
import pandas as pd
import asyncio
from example_analysis_pipeline import LangEvalAlgo

# Configuration
judge_model = "llama3.1:8b"
juror_model = "llama3.2:3b"
possible_emotions = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]
dataset_list = [
    "afr", "amh", "arq", "ary", "chn", "deu", "eng", "esp", "hau", "hin",
    "ibo", "kin", "mar", "orm", "pcm", "ptbr", "ptmz", "ron", "rus", "som",
    "sun", "swa", "swe", "tat", "tir", "ukr", "vmw", "yor",
]
batch_size = 50  # Number of rows to process concurrently

# Initialize the evaluator
evaluator = LangEvalAlgo(
    juror_model=juror_model,
    judge_model=judge_model,
    possible_emotions=possible_emotions
)

async def process_example(evaluator, example):
    """Process a single example asynchronously."""
    return await evaluator.run_jurors(example=example)

async def process_dataset(dataset):
    """Process an entire dataset asynchronously."""
    input_file_path = f"public_data_test/track_a/test/{dataset}.csv"
    output_dir = "results_complete"
    output_file_path = os.path.join(output_dir, f"{dataset}.csv")

    if not os.path.exists(input_file_path):
        print(f"Path not found: {input_file_path}")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(input_file_path)
    tasks = []
    buffer = []

    async with aiofiles.open(output_file_path, mode='a') as f:
        # Write header if file doesn't exist
        if not os.path.exists(output_file_path):
            header = ','.join(data.columns) + '\n'
            await f.write(header)

        for row in data.itertuples(index=False):
            example = row.text
            tasks.append(process_example(evaluator, example))

            # Process in batches
            if len(tasks) >= batch_size:
                responses = await asyncio.gather(*tasks)
                for response, row in zip(responses, data.itertuples(index=False)):
                    row_dict = row._asdict()
                    row_dict.update({
                        "language": response[0],
                        "juror_1": response[1],
                        "juror_2": response[2],
                        "juror_3": response[3],
                        "juror_4": response[4],
                    })
                    buffer.append(row_dict)

                # Write buffer to file
                for row in buffer:
                    line = ','.join(map(str, row.values())) + '\n'
                    await f.write(line)

                tasks = []
                buffer = []

        # Process any remaining tasks
        if tasks:
            responses = await asyncio.gather(*tasks)
            for response, row in zip(responses, data.itertuples(index=False)):
                row_dict = row._asdict()
                row_dict.update({
                    "language": response[0],
                    "juror_1": response[1],
                    "juror_2": response[2],
                        "juror_3": response[3],
                        "juror_4": response[4],
                    })
                buffer.append(row_dict)

                # Write remaining buffer to file
                for row in buffer:
                    line = ','.join(map(str, row.values())) + '\n'
                    await f.write(line)

    print(f"Completed processing for dataset: {dataset}")

async def main():
    """Main function to process all datasets."""
    tasks = [process_dataset(dataset) for dataset in dataset_list]
    await asyncio.gather(*tasks)

# Run the asynchronous processing
if __name__ == "__main__":
    asyncio.run(main())
