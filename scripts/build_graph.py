import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from mention_encoder import MentionEncoder
from tqdm import tqdm

MERGED_DATA_PATH = "data/merged_data.json"

def build_graph(use_cuda=False):
    """Builds a PyTorch Geometric graph from event mentions"""
    
    # Set device and print status
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"🔥 Building graph using device: {device}")
    
    # Ensure merged data file exists
    if not os.path.exists(MERGED_DATA_PATH):
        raise FileNotFoundError(f"❌ Missing file: {MERGED_DATA_PATH}")

    # Load merged data
    with open(MERGED_DATA_PATH, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    mentions = []
    encoder = MentionEncoder(device=device)  # Pass device to encoder
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

    # Process mentions with progress bar
    for m in tqdm(mentions, desc="Processing mentions"):
        embedding = encoder.encode(m["raw_text"], m["trigger_word"])
        if embedding is not None:
            # Keep embeddings on CPU until final conversion
            node_features.append(embedding.cpu().numpy())

    if not node_features:
        raise ValueError("❌ No valid embeddings found. Check mention extraction.")

    # Convert to tensor and move to device
    x = torch.tensor(np.vstack(node_features), dtype=torch.float).to(device)
    print(f"✅ Node features tensor shape: {x.shape}, device: {x.device}")

    num_nodes = len(mentions)
    src, dst, edge_labels = [], [], []

    # Create edges with progress bar
    print("Creating edges...")
    for i in tqdm(range(num_nodes), desc="Building edges"):
        for j in range(i + 1, num_nodes):
            src.append(i)
            dst.append(j)
            edge_labels.append(1)  # Dummy labels (modify based on dataset)

    # Move edge data to device
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(device)
    edge_labels_tensor = torch.tensor(edge_labels + edge_labels, dtype=torch.long).to(device)
    print(f"✅ Edge tensor shape: {edge_index.shape}, device: {edge_index.device}")

    data = Data(x=x, edge_index=edge_index)
    data.edge_labels = edge_labels_tensor
    
    # Print memory usage if using CUDA
    if use_cuda and torch.cuda.is_available():
        print(f"🔥 GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"🔥 GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    return data

if __name__ == "__main__":
    # Test with CUDA if available
    use_cuda = torch.cuda.is_available()
    graph_data = build_graph(use_cuda=use_cuda)
    print("✅ Graph built successfully!")
    print(f"Graph device: {graph_data.x.device}")
