import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple
import logging

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 128

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentionEncoder:
    def __init__(self, model_name=MODEL_NAME, device=None):
        """Initialize the encoder with specified device"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 MentionEncoder using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text, trigger_word):
        """Encodes a mention using LegalBERT"""
        if not text or not trigger_word:
            print(f"⚠️ Skipping empty mention: text='{text}', trigger_word='{trigger_word}'")
            return None

        inputs = self.tokenizer(f"{text} [SEP] {trigger_word}", 
                              return_tensors="pt", 
                              truncation=True, 
                              max_length=MAX_LENGTH)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # Return CPU tensor for numpy conversion
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

    def encode_batch(self, texts: list, trigger_words: list) -> list:
        """Encode multiple mentions in a batch
        
        Args:
            texts: List of texts containing mentions
            trigger_words: List of trigger words
            
        Returns:
            list: List of encoded mentions (None for failed encodings)
        """
        return [self.encode(text, trigger) for text, trigger in zip(texts, trigger_words)]

if __name__ == "__main__":
    encoder = MentionEncoder()
    sample_embedding = encoder.encode("Sample legal text", "OMITTED")
    print("✅ Embedding shape:", sample_embedding.shape)
    if torch.cuda.is_available():
        print("✅ CUDA is available and being used")
