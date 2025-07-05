import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from datasets import load_dataset
from dml.core import DMLTransformer
from dml.reward import RewardEvaluator
from torch.optim import AdamW
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loading
try:
    dataset = load_dataset("roneneldan/TinyStories", split="train").select(range(500))
    print("✅ Loaded TinyStories dataset (500 samples)")
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")
    dataset = [
        {"text": "Once upon a time there was a little dog named Max."},
        {"text": "The princess lived in a castle made of gold."},
        {"text": "Scientists discovered a new planet with oceans."}
    ]

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Model
model = DMLTransformer().to(device)
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

# Training settings
num_epochs = 10
batch_size = 4
novelty_weight = 0.05

# Metrics tracking
loss_history = []
novelty_history = []

# Batch creation
def create_batches(texts, batch_size):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_batches([example['text'] for example in dataset], batch_size)

# Reward evaluator
reward_evaluator = RewardEvaluator()

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_novelty = 0

    for batch_idx, (input_ids, attention_mask) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        optimizer.zero_grad()

        outputs = model(input_ids)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))

        # Novelty reward
        with torch.no_grad():
            current_embed = model.last_hidden.mean(dim=1)
            novelty = 1 - torch.nn.functional.cosine_similarity(current_embed, torch.randn_like(current_embed)).mean()
        loss -= novelty_weight * novelty.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_novelty += novelty.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Novelty: {novelty.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    avg_novelty = epoch_novelty / len(train_loader)
    loss_history.append(avg_loss)
    novelty_history.append(avg_novelty)

    print(f"\nEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f} | Avg Novelty: {avg_novelty:.4f}\n")

# Plot training progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(novelty_history, label='Novelty', color='orange')
plt.title('Novelty Score')
plt.xlabel('Epoch')
plt.ylabel('Novelty')
plt.tight_layout()
plt.show()