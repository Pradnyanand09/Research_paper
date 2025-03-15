import json
import os
import random

def set_seed(seed=42):
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_data(data_path, output_train, output_val, output_test, train_ratio=0.8, val_ratio=0.1):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    n = len(data)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Write JSONL files
    for split_path, split_data in zip([output_train, output_val, output_test], [train_data, val_data, test_data]):
        with open(split_path, "w", encoding="utf-8") as f:
            for entry in split_data:
                json.dump(entry, f)
                f.write('\n')
    print(f"Data split into {output_train}, {output_val}, and {output_test}")

if __name__ == "__main__":
    split_data(r"F:\reserach_paper_codes\data\merged_data.json",
               r"F:\reserach_paper_codes\data\train.jsonl",
               r"F:\reserach_paper_codes\data\val.jsonl",
               r"F:\reserach_paper_codes\data\test.jsonl")