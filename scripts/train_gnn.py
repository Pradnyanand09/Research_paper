import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from gnn_coref import CorefGAT
from build_graph import build_graph

CHECKPOINT_PATH = r"F:\reserach_paper_codes\models\saved_models\coref_gnn.pt"

def train_one_epoch(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    node_emb = model(data.x, data.edge_index)
    src, dst = data.edge_index[0], data.edge_index[1]
    logits = model.classify_edges(node_emb, src, dst)
    loss = F.cross_entropy(logits, data.edge_labels.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_graph()
    model = CorefGAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(num_epochs):
        loss_val = train_one_epoch(model, data, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ Model saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train_model()
