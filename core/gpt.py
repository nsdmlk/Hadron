import torch
import torch.nn as nn
from .transformer import Transformer
from .embedding import Embedding
from .bpe_tokenizer import BPETokenizer


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, d_ff=1024, n_layers=4):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_encoder_layers=n_layers,
            n_decoder_layers=n_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)  # предсказание токена
    
    def forward(self, token_ids):
        # token_ids: (batch, seq_len)
        x = self.embedding(token_ids)  # (batch, seq_len, d_model)
        output = self.transformer(x, x)  # (batch, seq_len, d_model)
        logits = self.lm_head(output)  # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, tokenizer, prompt, max_len=20, temperature=1.0):
        """Генерация текста по промпту."""
        self.eval()
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long)
        
        for _ in range(max_len):
            logits = self.forward(tokens)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        return tokenizer.decode(tokens[0].tolist())
    