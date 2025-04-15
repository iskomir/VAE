import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# Устанавливаем seed для воспроизводимости
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка данных
data, metadata = plasticc_gp()
data = data[:, :-11]
filtered_data = data[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]
filtered_metadata = metadata[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]

# VAE
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
        
        # Автоматический расчет выходного размера энкодера
        self.encoded_dim = input_dim
        for _ in range(3):
            self.encoded_dim = (self.encoded_dim + 1) // 2  # Для stride=2
        
        # Линейные слои для mu и logvar
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
            
            # Финальная свертка с padding для точного соответствия размеров
            nn.Conv1d(32, 1, kernel_size=3, padding=1, bias=True),
            nn.AdaptiveAvgPool1d(input_dim)  # Гарантируем точный размер выхода
        )
    
    def encode(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        h = self.encoder(x)  # [batch, 128, encoded_dim]
        h_flat = h.view(h.size(0), -1)  # [batch, 128 * encoded_dim]
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

# Функция потерь для VAE
def loss_function(recon_x, x, mu, logvar, kld_weight=0.5, recon_weight=1.0):
    BCE = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = kld_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Подготовка данных 
class MyDataset(Dataset):
    def __init__(self, filtered_data):
        self.data = filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Разбиваем данные на обучающую и валидационную выборки с сохранением пропорций аномалий
train_data, val_data, train_labels, val_labels = train_test_split(
    filtered_data, filtered_data, test_size=0.3, stratify=filtered_metadata, random_state=seed)

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Параметры модели
input_dim = data.shape[1]
# latent_dim = 2
# latent_dim = 4
# latent_dim = 8
latent_dim = 16

# Создание модели и перенос на GPU
model = VariationalAutoencoder(input_dim, latent_dim).to(device)

# Оптимизация
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 100
kld_weight = 0.5
recon_weight = 1.0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.6f}")

    # Оценка на валидационной выборке
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1}, Validation Loss: {val_loss:.6f}")

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Рассчет reconstruction errors для каждого образца
reconstruction_errors = []

model.eval()
with torch.no_grad():
    for sample in filtered_data:
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        output, mu, logvar = model(sample_tensor)
        reconstruction_error = nn.functional.mse_loss(output, sample_tensor).item()
        reconstruction_errors.append(reconstruction_error)

# Преобразуем в массив NumPy для удобства
reconstruction_errors = np.array(reconstruction_errors)

# Сортировка ошибок и выбор первых 100
sorted_indices = np.argsort(reconstruction_errors)[::-1]  # Сортируем по убыванию
top_100_indices = sorted_indices[:100]
top_100_errors = reconstruction_errors[top_100_indices]

# Получаем соответствующие объекты и метки
top_100_objects = filtered_data[top_100_indices]
top_100_labels = filtered_metadata[top_100_indices]

# Вывод результатов
print("Индексы 100 образцов с наибольшей реконструкционной ошибкой:", top_100_indices)
print("Реконструкционные ошибки для этих образцов:", top_100_errors)
print("Метки для этих образцов:", top_100_labels)

plt.figure(figsize=(8, 4))
plt.hist(reconstruction_errors, bins=30, density=True, alpha=0.6, color='g')
plt.title(f'Гистограмма RE')
plt.xlabel('reconstruction_error')
plt.ylabel('Плотность')
plt.grid(True)
plt.savefig('reconstruction_errors.pdf', format='pdf', bbox_inches='tight')
plt.close()
