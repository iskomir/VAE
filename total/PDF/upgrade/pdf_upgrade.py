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
from scipy.stats import multivariate_normal

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

# Подготовка данных (обучаем на всех данных)
train_data, val_data = train_test_split(filtered_data, test_size=0.3, random_state=seed)
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Инициализация модели
input_dim = data.shape[1]
# latent_dim = 2  
# latent_dim = 4
# latent_dim = 8
latent_dim = 16 
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

# Получение латентных представлений
latent_representations = []
with torch.no_grad():
    for sample in filtered_data:
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        mu, _ = model.encode(sample_tensor)
        latent_representations.append(mu.cpu().numpy())

latent_representations = np.array(latent_representations).squeeze()

# Вычисление PDF только на нормальных данных
normal_latent = latent_representations[filtered_metadata == 1]
mean = np.mean(normal_latent, axis=0)
cov = np.cov(normal_latent, rowvar=False)

# Вычисление плотности вероятности для всех точек
try:
    pdf_values = multivariate_normal.pdf(latent_representations, mean=mean, cov=cov, allow_singular=True)
except:
    # Если матрица ковариации вырождена, добавляем небольшой шум
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    pdf_values = multivariate_normal.pdf(latent_representations, mean=mean, cov=cov, allow_singular=True)

# Определение аномалий (5% наименьших плотностей)
threshold = np.percentile(pdf_values, 5)
predictions = np.where(pdf_values < threshold, -1, 1) # 1 - нормальные, -1 - аномалии
true_labels = np.where(filtered_metadata == 1, 1, -1) # делаем аномалии из 0 в -1

# Метрики качества
accuracy = accuracy_score(true_labels, predictions)
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['Anomaly', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions))
print(f"\nAccuracy: {accuracy:.4f}")

# Топ-100 аномалий
top_100_indices = np.argsort(pdf_values)[:100]
top_100_pdfs = pdf_values[top_100_indices]
top_100_labels = filtered_metadata[top_100_indices]

print("\nTop 100 anomalies:")
for i in range(100):
    print(f"Index: {top_100_indices[i]}, PDF: {top_100_pdfs[i]}, Label: {top_100_labels[i]}")

# Визуализация
plt.figure(figsize=(15, 5))

# 1. Кривая обучения
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Curve')

# 2. Распределение PDF значений
plt.subplot(1, 3, 2)
plt.hist(pdf_values[filtered_metadata == 1], bins=50, alpha=0.5, label='Normal')
plt.hist(pdf_values[filtered_metadata == 0], bins=50, alpha=0.5, label='Anomaly')
plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.xlabel('PDF Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('PDF Values Distribution')

# 3. Латентное пространство (первые 2 измерения)
plt.subplot(1, 3, 3)
plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
            c=np.where(filtered_metadata == 1, 'b', 'r'), alpha=0.3, s=5)
plt.scatter(latent_representations[top_100_indices, 0], 
            latent_representations[top_100_indices, 1],
            c='yellow', edgecolor='k', s=50, label='Top Anomalies')
plt.xlabel('Latent Dim 1')
plt.ylabel('Latent Dim 2')
plt.legend()
plt.title('Latent Space')

plt.tight_layout()
plt.savefig('vae_pdf_results.pdf', format='pdf')
plt.show()