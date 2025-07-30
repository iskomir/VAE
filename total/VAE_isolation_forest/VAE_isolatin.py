import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from plasticc_gp import plasticc_gp

# Установка seed для воспроизводимости
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Устройство для обучения 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка и фильтрация данных
data, metadata = plasticc_gp()
data = data[:, :-11]  # Удаление последних 11 столбцов
filter_mask = (data >= -1).all(axis=1) & (data <= 1).all(axis=1)
filtered_data = data[filter_mask]
filtered_metadata = metadata[filter_mask]

# Архитектура VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Энкодер
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
        
        # Вычисление размерности после сверток
        self.encoded_dim = input_dim
        for _ in range(3):
            self.encoded_dim = (self.encoded_dim + 1) // 2
            
        # Полносвязные слои для mu и logvar
        self.fc_mu = nn.Linear(128 * self.encoded_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * self.encoded_dim, latent_dim)
        
        # Декодер
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z).squeeze(1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Функция потерь для VAE
def loss_function(recon_x, x, mu, logvar, kld_weight=0.5, recon_weight=1.0):
    BCE = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = kld_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Датасет
class AnomalyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Подготовка данных
train_data, test_data, train_meta, test_meta = train_test_split(
    filtered_data, filtered_metadata, test_size=0.3, random_state=seed
)

train_dataset = AnomalyDataset(train_data)
test_dataset = AnomalyDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Инициализация и обучение VAE
input_dim = filtered_data.shape[1]
# latent_dim = 2
# latent_dim = 4
# latent_dim = 8  
latent_dim = 16

model = VariationalAutoencoder(input_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение VAE
epochs = 50
train_losses = []
val_losses = []

print("Training VAE...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            val_loss += loss_function(recon_batch, batch, mu, logvar).item()
    
    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Получение латентных представлений для всех данных
with torch.no_grad():
    latent_representations = []
    for sample in filtered_data:
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        mu, _ = model.encode(sample_tensor)
        latent_representations.append(mu.cpu().numpy())
latent_representations = np.array(latent_representations).squeeze()

# Подготовка меток (1 - нормальные, -1 - аномалии)
true_labels = np.where(filtered_metadata == 1, 1, -1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    latent_representations, true_labels, test_size=0.3, random_state=seed
)

# Обучение Isolation Forest на латентных представлениях
print("\nTraining Isolation Forest on latent space...")
iso_forest = IsolationForest(
    n_estimators=300,
    max_samples=256,
    contamination='auto',
    random_state=seed,
    n_jobs=-1,
    verbose=1
)
iso_forest.fit(X_train)

# Предсказания на тестовых данных
test_pred = iso_forest.predict(X_test)

# Метрики качества
accuracy = accuracy_score(y_test, test_pred)
print("\nClassification Report (Isolation Forest on Latent Space):")
print(classification_report(y_test, test_pred, target_names=['Anomaly', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print(f"\nAccuracy: {accuracy:.4f}")

# Визуализация результатов
plt.figure(figsize=(15, 5))

# 1. Кривая обучения VAE
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Распределение anomaly scores
plt.subplot(1, 3, 2)
all_scores = iso_forest.decision_function(latent_representations)

n_normal = sum(filtered_metadata == 1)
n_anomaly = sum(filtered_metadata != 1)

plt.hist(all_scores[filtered_metadata == 1], 
         bins=50, alpha=0.5, label=f'Normal (n={n_normal})', color='blue')
plt.hist(all_scores[filtered_metadata != 1], 
         bins=50, alpha=0.5, label=f'Anomaly (n={n_anomaly})', color='red')

plt.xlabel('Anomaly Score')
plt.ylabel('Frequency')
plt.title('Anomaly Scores Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Латентное пространство с аномалиями (первые 2 измерения)
plt.subplot(1, 3, 3)
top_100_indices = np.argsort(all_scores)[:100]  # Наименьшие scores - самые аномальные

plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
           c=np.where(filtered_metadata == 1, 'blue', 'red'), 
           alpha=0.3, s=5)
plt.scatter(latent_representations[top_100_indices, 0], 
           latent_representations[top_100_indices, 1],
           c='yellow', edgecolor='k', s=50, label='Top 100 Anomalies')

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space with Anomalies')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_isolation_forest_results_dim16.pdf', format='pdf', bbox_inches='tight')
print("\nResults saved to 'vae_isolation_forest_results.pdf'")