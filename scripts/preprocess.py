import os
import json

# Define file paths
DATA_JSONL_PATH = r"F:\reserach_paper_codes\data\data.jsonl"
RAW_DATA_DIR = r"F:\reserach_paper_codes\data\raw_data"
OUTPUT_JSON = r"F:\reserach_paper_codes\data\merged_data.json"

def preprocess_data():
    """
    Merges JSONL annotations with raw text files and saves as merged_data.json.
    """

    # Check if the annotation file exists
    if not os.path.exists(DATA_JSONL_PATH):
        raise FileNotFoundError(f"❌ JSONL file not found: {DATA_JSONL_PATH}")

    data_entries = []
    
    # Read JSONL file (one JSON object per line)
    with open(DATA_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Avoid empty lines
                try:
                    entry = json.loads(line)
                    data_entries.append(entry)
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON entry: {line}")

    merged_dataset = []
    missing_files = 0  # Counter for missing raw text files

    for entry in data_entries:
        doc_id = entry.get("id")

        # Ensure doc_id is a string (some datasets use int IDs)
        doc_id_str = str(doc_id)  # Convert to string for filename matching

        # Generate the expected raw text file path
        raw_file_path = os.path.join(RAW_DATA_DIR, f"{doc_id_str}.txt")  # Matches 1.txt, 2.txt, etc.

        if not os.path.exists(raw_file_path):
            print(f"⚠️ Missing file for doc {doc_id}. Expected: {raw_file_path}")
            missing_files += 1
            continue

        # Read the raw text file
        with open(raw_file_path, "r", encoding="utf-8") as rf:
            raw_text = rf.read()

        # Create a structured merged record
        merged_record = {
            "id": doc_id,
            "raw_text": raw_text,
            "events": entry.get("events", [])  # Get events or default to empty list
        }
        merged_dataset.append(merged_record)

    # Save the merged dataset to a JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
        json.dump(merged_dataset, out_f, indent=2)

    print(f"✅ Preprocessing complete. Merged {len(merged_dataset)} documents.")
    if missing_files > 0:
        print(f"⚠️ {missing_files} documents were skipped due to missing raw text files.")

if __name__ == "__main__":
    preprocess_data()
