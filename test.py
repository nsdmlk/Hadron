import torch
from core.transformer import Transformer
from core.mask import generate_mask

model = Transformer(d_model=512, n_heads=8, d_ff=2048, n_encoder_layers=6, n_decoder_layers=6)

src = torch.randn(2, 10, 512)  # вход (encoder)
tgt = torch.randn(2, 15, 512)  # выход (decoder)

src_mask = None
tgt_mask = generate_mask(15)

out = model(src, tgt, src_mask, tgt_mask)
print(out.shape)  # (2, 15, 512)