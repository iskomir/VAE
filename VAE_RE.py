import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
import numpy as np
import matplotlib.pyplot as plt
import random

# Уменьшаем вариативность весов
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Загрузка данных
data, metadata = plasticc_gp()
data = data[:, :-11]
filtered_data = data[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]

# Определение VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(2, latent_dim)
        self.logvar_layer = nn.Linear(2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 7), 
            nn.ReLU(),
            nn.Linear(7, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Функция потерь для VAE
def loss_function(recon_x, x, mu, logvar, kld_weight=1.2, recon_weight=0.6):
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

# Разделение данных на обучающую и валидационную выборки
train_data = filtered_data[:int(0.7 * len(data))]
val_data = filtered_data[int(0.3 * len(data)):]

train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size=22, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=22, shuffle=False)

# Параметры модели
input_dim = data.shape[1]
latent_dim = 2

# Создание модели
model = VariationalAutoencoder(input_dim, latent_dim)

# Оптимизация
optimizer = optim.SGD(model.parameters(), lr=0.000025)

# Обучение модели
epochs = 50
kld_weight = 1.5
recon_weight = 1.0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    train_loss /= len(train_dataloader.dataset)
    train_losses.append(train_loss)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss}")

    # Оценка на валидационной выборке
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar, kld_weight, recon_weight)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}")

# Построение графика потерь
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Рассчет reconstruction errors для каждого образца
reconstruction_errors = []

model.eval()
with torch.no_grad():
    for sample in data:
        # Проверка на наличие nan или inf в данных
        if np.isnan(sample).any() or np.isinf(sample).any():
            print("Обнаружены некорректные значения в данных. Пропускаем этот образец.")
            continue
        
        # Проверка на корректность данных 
        if not (np.all(sample >= -1) and np.all(sample <= 1)):
            print("Обнаружены данные вне допустимого диапазона. Пропускаем этот образец.")
            continue
        
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        output, mu, logvar = model(sample_tensor)
        
        # Проверка на наличие nan или inf в выходных данных
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Обнаружены некорректные значения в выходных данных. Пропускаем этот образец.")
            continue
        
        reconstruction_error = nn.functional.mse_loss(output, sample_tensor).item()
        reconstruction_errors.append(reconstruction_error)

# Преобразуем в массив NumPy для удобства
reconstruction_errors = np.array(reconstruction_errors)

# Сортировка ошибок и выбор первых 100
sorted_indices = np.argsort(reconstruction_errors)[::-1]  # Сортируем по убыванию
top_100_indices = sorted_indices[:100]
top_100_errors = reconstruction_errors[top_100_indices]

# Вывод результатов
print("Индексы 100 образцов с наибольшей реконструкционной ошибкой:", top_100_indices)
print("Реконструкционные ошибки для этих образцов:", top_100_errors)

plt.figure(figsize=(8, 4))
plt.hist(reconstruction_errors, bins=30, density=True, alpha=0.6, color='g')
plt.title(f'Гистограмма для латентного слоя')
plt.xlabel('Значение')
plt.ylabel('Плотность')
plt.grid(True)
plt.show()
