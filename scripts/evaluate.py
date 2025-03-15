import torch
import torch.nn.functional as F
from gnn_coref import CorefGAT
from build_graph import build_graph

def evaluate_model(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        node_emb = model(data.x, data.edge_index)
        src = data.edge_index[0]
        dst = data.edge_index[1]
        logits = model.classify_edges(node_emb, src, dst)
        preds = logits.argmax(dim=1)
        labels = data.edge_labels.to(device)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = build_graph(r"F:\reserach_paper_codes\data\test.jsonl")  # Use test split
    model = CorefGAT().to(device)
    checkpoint_path = r"F:\reserach_paper_codes\models\saved_models\coref_gnn.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    acc = evaluate_model(model, data, device)
    print(f"Evaluation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()