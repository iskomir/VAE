import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from plasticc_gp import plasticc_gp
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# Уменьшаем вариативность весов
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Загрузка данных
data, metadata = plasticc_gp()
data = data[:, :-11]
filtered_data = data[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]
filtered_metadata = metadata[(data >= -1).all(axis=1) & (data <= 1).all(axis=1)]

# Определение VAE
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU()
        )
        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64), 
            nn.ReLU(),
            # nn.Linear(32, 64), 
            # nn.ReLU(),
            nn.Linear(64, input_dim)
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
def loss_function(recon_x, x, mu, logvar, kld_weight=0.5, recon_weight=1.0):
     # Проверка logvar
    if torch.isnan(logvar).any() or torch.isinf(logvar).any():
        print("Обнаружены некорректные значения в logvar.")
        return torch.tensor(0.0, requires_grad=True)  # Возвращаем нулевой тензор
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

# Создание модели
model = VariationalAutoencoder(input_dim, latent_dim)

# Оптимизация
# optimizer = optim.SGD(model.parameters(), lr=0.000025)
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

# Получение скрытых представлений
latent_representations = []

model.eval()
with torch.no_grad():
    for sample in filtered_data:
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        mu, logvar = model.encode(sample_tensor)
        latent_representations.append(mu.numpy())  # Преобразуем в NumPy массив

# Преобразуем список в двумерный массив NumPy
latent_representations = np.array(latent_representations).squeeze()  # Удаляем лишние размерности

# Параметры многомерного нормального распределения
mean = np.mean(latent_representations, axis=0)
cov = np.cov(latent_representations, rowvar=False)

# Вычисление плотности вероятности
pdf_values = multivariate_normal.pdf(latent_representations, mean=mean, cov=cov, allow_singular=True)

# Сортировка по убыванию и выбор 100 самых маленьких
sorted_indices = np.argsort(pdf_values)  # Сортируем по возрастанию
top_100_indices = sorted_indices[:100]
top_100_pdf_values = pdf_values[top_100_indices]

# Получаем соответствующие объекты и метки
top_100_objects = filtered_data[top_100_indices]
top_100_labels = filtered_metadata[top_100_indices]

# Вывод результатов
print("Индексы 100 образцов с наименьшей плотностью вероятности:", top_100_indices)
print("Плотности вероятности для этих образцов:", top_100_pdf_values)
print("Метки для этих образцов:", top_100_labels)

# Построение гистограммы для плотности распределения (должен быть пик)
plt.figure(figsize=(8, 4))
plt.hist(pdf_values, bins=30, density=True, alpha=0.7, color='blue')
plt.title('PDF')
plt.xlabel('pdf_values')
plt.ylabel('Плотность')
plt.grid()
plt.show()
