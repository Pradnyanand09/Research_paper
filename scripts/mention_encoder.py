import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

class MentionEncoder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text, trigger_word):
        """Encodes a mention using LegalBERT"""
        if not text or not trigger_word:
            print(f"⚠️ Skipping empty mention: text='{text}', trigger_word='{trigger_word}'")
            return None

        inputs = self.tokenizer(f"{text} [SEP] {trigger_word}", return_tensors="pt", truncation=True, max_length=128)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

if __name__ == "__main__":
    encoder = MentionEncoder()
    sample_embedding = encoder.encode("Sample legal text", "OMITTED")
    print("✅ Embedding shape:", sample_embedding.shape)
