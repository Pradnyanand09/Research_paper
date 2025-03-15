import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from mention_encoder import MentionEncoder

MERGED_DATA_PATH = r"F:\reserach_paper_codes\data\merged_data.json"

def build_graph():
    """Builds a PyTorch Geometric graph from event mentions"""

    # Ensure merged data file exists
    if not os.path.exists(MERGED_DATA_PATH):
        raise FileNotFoundError(f"❌ Missing file: {MERGED_DATA_PATH}")

    # Load merged data
    with open(MERGED_DATA_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    mentions = []
    encoder = MentionEncoder()
    node_features = []

    for record in merged_data:
        doc_id = record["id"]
        raw_text = record["raw_text"]

        for event in record.get("events", []):  # Safely get events
            for mention in event.get("mention", []):
                mention_record = {
                    "doc_id": doc_id,
                    "trigger_word": mention["trigger_word"],
                    "raw_text": raw_text
                }
                mentions.append(mention_record)

    print(f"🔍 Found {len(mentions)} mentions to process.")

    for m in mentions:
        embedding = encoder.encode(m["raw_text"], m["trigger_word"])
        if embedding is not None:
            node_features.append(embedding.numpy())

    if not node_features:
        raise ValueError("❌ No valid embeddings found. Check mention extraction.")

    x = torch.tensor(np.vstack(node_features), dtype=torch.float)

    num_nodes = len(mentions)
    src, dst, edge_labels = [], [], []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            src.append(i)
            dst.append(j)
            edge_labels.append(1)  # Dummy labels (modify based on dataset)

    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_labels_tensor = torch.tensor(edge_labels + edge_labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.edge_labels = edge_labels_tensor
    return data

if __name__ == "__main__":
    graph_data = build_graph()
    print("✅ Graph built successfully!")
