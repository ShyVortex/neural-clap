import json
import os

import pandas as pd
import requests

this_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.normpath(os.path.join(this_dir, '..', 'data', 'rq1-data-nc.csv'))
text_column = 'review'
output_path = os.path.normpath(os.path.join(this_dir, '..', 'output'))
output_file = os.path.join(output_path, 'classified_reviews.jsonl')

model = 'llama3.2'
ollama_api = "http://localhost:11434/api/generate"

zshot_path = os.path.normpath(os.path.join(this_dir, '..', 'prompts', 'zero_shot.txt'))
fshot_path = os.path.normpath(os.path.join(this_dir, '..', 'prompts', 'few_shot.txt'))
cot_path = os.path.normpath(os.path.join(this_dir, '..', 'prompts', 'chain_of_thought.txt'))


def get_processed_count():
    """Counts how many rows have been written in the output file."""
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        return 0
    with open(output_file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def call_ollama(text):
    """Sends request to Ollama and returns response."""
    zshot_prompt = open(zshot_path, 'r').read()
    fshot_prompt = open(fshot_path, 'r').read()
    cot_prompt = open(cot_path, 'r').read()

    input_prompt = f"Categorize: {text}"

    payload = {
        "model": model,
        "prompt": input_prompt,
        "system": zshot_prompt,
        "stream": False,  # False = we wait for the complete answer,
        "options": {
            "temperature": 0.0,  # low temperature leads to more technical answers,
            "seed": 123,         # for reproducibility
            "num_predict": 5,
            "num_ctx": 4096,     # enough context for a review
            "stop": ["\n", "."], # stop the model on new line or period
            "top_k": 10,         # reduce probability of generating gibberish
            "top_p": 0.5,        # value for more or less varied output
        },
    }

    try:
        response = requests.post(ollama_api, json=payload)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None


def main():
    # 1. Dataset loading
    print(f"Loading data from {input_file}...")
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            print("Unsupported file format. Use CSV or XLSX.")
            return
    except Exception as e:
        print(f"Error while opening file: {e}")
        return

    total_rows = len(df)

    # 2. Resume check
    processed_count = get_processed_count()

    if processed_count > 0:
        print(f"Found pre-existing output file with {processed_count} instances.")
        if processed_count >= total_rows:
            print("All rows have already been analyzed!")
            return
        print(f"Resuming analysis from row {processed_count + 1}...")
    else:
        print("No output file found. Starting from scratch...")

    # 3. Processing loop
    ## Dataframe slicing: we skip rows already processed
    df_to_process = df.iloc[processed_count:]

    valid_answers = ["BUG", "FEATURE", "SECURITY", "PERFORMANCE", "USABILITY", "ENERGY", "OTHER"]

    for index, row in df_to_process.iterrows():
        current_step = index + 1
        text_content = str(row[text_column])

        print(f"Analysis in progress... [{current_step}/{total_rows}]", end="\r")

        llm_response = call_ollama(text_content)

        if llm_response:
            clean_category = llm_response.upper().replace('.', '').strip()

            # If result is not valid, we classify it as ERROR
            final_category = clean_category if clean_category in valid_answers else "ERROR"

            result_obj = {
                "id": current_step,
                "text": text_content,
                "category": final_category
            }

            # 4. We append the result immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result_obj, f, ensure_ascii=False)
                f.write('\n')  # new row for JSONL format
        else:
            print(f"\nError thrown in line {current_step}, row skipped.")

    print(f"\nAnalysis complete! Data has been saved to {output_file}")


if __name__ == "__main__":
    main()