import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class CorefGAT(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_heads=4):
        super(CorefGAT, self).__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=True, dropout=0.1)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

    def classify_edges(self, node_emb, src, dst):
        edge_features = torch.cat([node_emb[src], node_emb[dst]], dim=1)
        logits = self.edge_classifier(edge_features)
        return logits

if __name__ == "__main__":
    import torch
    x = torch.randn(10, 768)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    model = CorefGAT()
    node_emb = model(x, edge_index)
    print("Node embeddings shape:", node_emb.shape)