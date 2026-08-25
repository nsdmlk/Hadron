import subprocess
import torch
import torch.nn as nn
from core.gpt import GPT
from core.bpe_tokenizer import BPETokenizer

# 1. Получить git help как данные
result = subprocess.run(['git', '--help'], capture_output=True, text=True)
text = result.stdout + result.stderr

# Плюс выводы всех команд
commands = ['clone', 'init', 'add', 'commit', 'push', 'pull', 'branch', 'merge', 
            'log', 'diff', 'status', 'fetch', 'reset', 'rebase']
for cmd in commands:
    r = subprocess.run(['git', cmd, '--help'], capture_output=True, text=True)
    text += r.stdout + r.stderr

# Разбить на строки/слова
texts = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
print(f"Loaded {len(texts)} lines, {len(text)} chars")

# 2. Токенизатор
tokenizer = BPETokenizer()
tokenizer.train(texts, num_merges=100)

# 3. Кодируем
encoded_texts = [tokenizer.encode(t) for t in texts if len(t.split()) > 2]

# 4. Модель
vocab_size = len(tokenizer.vocab)
model = GPT(vocab_size=vocab_size, d_model=128, n_heads=4, d_ff=512, n_layers=2)

# 5. Оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 6. Обучение
for epoch in range(100):
    total_loss = 0
    count = 0
    for tokens in encoded_texts:
        if len(tokens) < 3:
            continue
        x = torch.tensor([tokens[:-1]], dtype=torch.long)
        y = torch.tensor([tokens[1:]], dtype=torch.long)
        
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / max(count, 1):.4f}")

# 7. Генерация
prompt = "git clone"
generated = model.generate(tokenizer, prompt, max_len=10)
print(f"Generated: {generated}")