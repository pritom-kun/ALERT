from torch import nn
import torch
import torch.nn.functional as F

class DropoutHead(nn.Module):
    """Custom classification head with dropout for uncertainty estimation"""
    def __init__(self, input_size, num_classes, dropout_rate=0.9):
        super(DropoutHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

def train_dropout(embeddings, labels, num_classes, batch_size, epochs, device):

    model = DropoutHead(768, num_classes, dropout_rate=0.9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    train_loss = 0
    num_samples = 0
    for _ in range(epochs):
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device).long()

            optimizer.zero_grad()
            logits = model(batch_embeddings)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_samples += batch_size

    print(f"====> Dropout Head Training Average loss: {train_loss / num_samples:.4f}")

    return model
