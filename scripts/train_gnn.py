import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gnn_coref import CorefGAT
from build_graph import build_graph
from typing import Optional
import logging
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/saved_models/coref_gnn.pt"
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 32

def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def train_one_epoch(model: CorefGAT, data: torch.Tensor, optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train the model for one epoch
    
    Args:
        model: The GNN model
        data: The graph data
        optimizer: The optimizer
        device: The device to train on
        
    Returns:
        float: The average loss for this epoch
    """
    model.train()
    total_loss = 0
    
    # Move data to device once
    data = data.to(device)
    logger.info(f"Data moved to device: {data.x.device}")
    print_gpu_memory()
    
    # Create batches of edges
    edge_indices = torch.arange(data.edge_index.size(1))
    batches = torch.split(edge_indices, BATCH_SIZE)
    
    for batch_idx in tqdm(batches, desc="Training batch"):
        optimizer.zero_grad()
        
        # Get node embeddings
        node_emb = model(data.x, data.edge_index)
        
        # Get edge indices for this batch
        src = data.edge_index[0][batch_idx].to(device)
        dst = data.edge_index[1][batch_idx].to(device)
        
        # Compute logits and loss for this batch
        logits = model.classify_edges(node_emb, src, dst)
        loss = F.cross_entropy(logits, data.edge_labels[batch_idx].to(device))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Clear GPU memory after each batch if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / len(batches)

def train_model(
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    use_cuda: bool = True,
    checkpoint_path: Optional[str] = CHECKPOINT_PATH
) -> None:
    """Train the GNN model
    
    Args:
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimization
        use_cuda: Whether to use CUDA if available
        checkpoint_path: Path to save the model checkpoint
    """
    # Set device
    if use_cuda and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU instead.")
        use_cuda = False
    
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    if use_cuda:
        # Print CUDA device properties
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        print_gpu_memory()
    
    # Build graph and create model
    logger.info("Building graph...")
    data = build_graph(use_cuda=use_cuda)
    logger.info(f"Graph built on device: {data.x.device}")
    
    model = CorefGAT().to(device)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        loss_val = train_one_epoch(model, data, optimizer, device)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Loss: {loss_val:.4f}")
        logger.info(f"Time: {epoch_time:.2f}s")
        print_gpu_memory()
        
        # Save best model
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✅ Saved best model with loss: {best_loss:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_model(use_cuda=True)  # Explicitly enable CUDA
