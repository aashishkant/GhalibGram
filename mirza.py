import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding


with open("mirza.txt", "r", encoding="utf-8") as file:
    data = file.read()

def get_char_mapping(data):
    unique_chars = list(set(data))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for char, idx in char_to_index.items()}
    return char_to_index, index_to_char

char_to_index, index_to_char = get_char_mapping(data)

block_size = 64
batch_size = 16
max_iters = 5000
learning_rate = 1e-4
eval_iters = 250
temperature = 1.0
device = torch.device("cpu")

# Assuming 'data' is defined elsewhere in your code
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([torch.tensor([char_to_index[char] for char in data_split[i:i + block_size]]) for i in ix])
    y = torch.stack([torch.tensor([char_to_index[char] for char in data_split[i + 1:i + block_size + 1]]) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(BigramLanguageModel, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.gru = GRU(embedding_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, targets=None):
        embedded = self.embedding(input)
        gru_out, _ = self.gru(embedded)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        else:
            loss = None

        return logits, loss

    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)

            # Fixing the dimension mismatch
            index_next = index_next.view(-1, 1)

            input = torch.cat((input, index_next), dim=1)
        return input

# Define your vocabulary size based on the content of the Oscar text file
vocab_size = len(set(data))
embedding_size = 128
hidden_size = 256

model = BigramLanguageModel(vocab_size, embedding_size, hidden_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
generated_chars = ''.join([index_to_char[idx] for idx in generated_indices])
print(generated_chars)
