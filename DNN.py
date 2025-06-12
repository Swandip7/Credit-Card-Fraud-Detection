import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=30, encoding_dim=14, hidden_layers=[64, 32], dropout_rate=0.2):
        super(AutoEncoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, encoding_dim))
        layers.append(nn.ReLU())

        for h_dim in reversed(hidden_layers):
            layers.append(nn.Linear(encoding_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            encoding_dim = h_dim
      
        layers.append(nn.Linear(prev_dim, input_dim))
        self.autoencoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.autoencoder(x)
input_dim = 30
encoding_dim = 14
hidden_layers = [64, 32]
dropout_rate = 0.2
learning_rate = 0.001
batch_size = 32
epochs = 50

model = AutoEncoder(input_dim, encoding_dim, hidden_layers, dropout_rate)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:  
        inputs = data.float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
