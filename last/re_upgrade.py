import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Устанавливаем seed для воспроизводимости
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка данных
data, metadata = plasticc_gp()
data = data[:, :-11]
filtered_data = data[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]
filtered_metadata = metadata[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]

# Архитектура VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        
        self.encoded_dim = input_dim
        for _ in range(3):
            self.encoded_dim = (self.encoded_dim + 1) // 2
            
        self.fc_mu = nn.Linear(128 * self.encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * self.encoded_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.encoded_dim),
            nn.Unflatten(1, (128, self.encoded_dim)),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1, bias=True),
            nn.AdaptiveAvgPool1d(input_dim)
        )
    
    def encode(self, x):
        x = x.unsqueeze(1)
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z).squeeze(1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, kld_weight=0.5, recon_weight=1.0):
    BCE = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = kld_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Подготовка данных
train_data, val_data = train_test_split(filtered_data, test_size=0.3, random_state=seed)
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Инициализация модели
input_dim = data.shape[1]
latent_dim = 2
# latent_dim = 4
# latent_dim = 8
# latent_dim = 16
model = VariationalAutoencoder(input_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            val_loss += loss_function(recon_batch, batch, mu, logvar).item()
    
    val_loss /= len(val_dataloader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Вычисление Reconstruction Error
reconstruction_errors = []
with torch.no_grad():
    for sample in filtered_data:
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        output, mu, logvar = model(sample_tensor)
        reconstruction_errors.append(nn.functional.mse_loss(output, sample_tensor).item())

reconstruction_errors = np.array(reconstruction_errors)

# Определение аномалий
threshold = np.percentile(reconstruction_errors, 95)
predictions = np.where(reconstruction_errors > threshold, -1, 1)
true_labels = np.where(filtered_metadata == 1, 1, -1)

# Метрики качества
accuracy = accuracy_score(true_labels, predictions)
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['Anomaly', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions))
print(f"\nAccuracy: {accuracy:.4f}")

# Топ-100 аномалий
top_100_indices = np.argsort(reconstruction_errors)[:100]
top_100_pdfs = reconstruction_errors[top_100_indices]
top_100_labels = filtered_metadata[top_100_indices]

print("\nTop 100 anomalies:")
for i in range(100):
    print(f"Index: {top_100_indices[i]}, PDF: {top_100_pdfs[i]}, Label: {top_100_labels[i]}")

# Визуализация
plt.figure(figsize=(12, 5)) 

# 1. Кривая обучения 
plt.subplot(1, 2, 1)  
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', linewidth=2, color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss', linewidth=2, color='orange')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.title('Training Curve', fontsize=14)
plt.grid(True, alpha=0.3)

# 2. Распределение ошибок реконструкции 
plt.subplot(1, 2, 2)
# Автоматическое определение границ по квантилям
re_min, re_max = np.quantile(reconstruction_errors, [0.05, 0.95])
re_pad = (re_max - re_min) * 0.1

plt.hist(reconstruction_errors[filtered_metadata == 1], 
         bins=np.linspace(re_min-re_pad, re_max+re_pad, 50),
         alpha=0.5, label='Normal', color='blue')
plt.hist(reconstruction_errors[filtered_metadata != 1],
         bins=np.linspace(re_min-re_pad, re_max+re_pad, 50),
         alpha=0.5, label='Anomaly', color='red')

plt.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
            label=f'Threshold: {threshold:.4f}')

plt.xlim(re_min-re_pad, re_max+re_pad)
plt.xlabel('Reconstruction Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=10)
plt.title('Reconstruction Error Distribution', fontsize=14)
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('vae_re_results.pdf', format='pdf')
