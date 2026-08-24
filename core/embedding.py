import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, token_ids):
        return self.embedding(token_ids)

embedding = Embedding(vocab_size=1000, d_model=512)
token_ids = torch.tensor([[5, 10, 15], [20, 25, 30]])  # 2 предложения по 3 слова
vectors = embedding(token_ids)
print(vectors.shape)  # (2, 3, 512)