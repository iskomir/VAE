import numpy as np
from sklearn.ensemble import IsolationForest
from plasticc_gp import plasticc_gp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Загрузка и фильтрация данных
data, metadata = plasticc_gp()
data = data[:, :-11]  
filter_mask = (data >= -1).all(axis=1) & (data <= 1).all(axis=1)
filtered_data = data[filter_mask]
filtered_metadata = metadata[filter_mask]

# Разделение на обучающую и тестовую выборки (без пропорциональности по аномалиям)
X_train, X_test, y_train, y_test = train_test_split(
    filtered_data, filtered_metadata, test_size=0.3, random_state=42
)

# Создание и обучение модели Isolation Forest
iso_forest = IsolationForest(
    n_estimators=300,          # Количество деревьев
    max_samples=256,           # Фиксированный размер подвыборки
    contamination='auto',      # Автоматическая оценка доли аномалий
    random_state=42,
    verbose=1,
    n_jobs=-1                 # Использование всех ядер процессора
)

# Обучение модели 
iso_forest.fit(X_train)

# Предсказания
test_pred = iso_forest.predict(X_test)
y_test_iso = np.where(y_test == 1, 1, -1)

# Метрики качества
accuracy = accuracy_score(y_test_iso, test_pred)
print("\nClassification Report:")
print(classification_report(y_test_iso, test_pred, target_names=['Anomaly', 'Normal']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_iso, test_pred))
print(f"\nAccuracy: {accuracy:.4f}")

# Полные предсказания для всех данных
all_pred = iso_forest.predict(filtered_data)
all_scores = iso_forest.decision_function(filtered_data)

# Нахождение топ-100 аномалий (с наименьшими scores)
top_100_indices = np.argsort(all_scores)[:100]
top_100_scores = all_scores[top_100_indices]
top_100_labels = filtered_metadata[top_100_indices]

print("\nTop 100 anomalies:")
for i in range(100):
    print(f"Index: {top_100_indices[i]}, Score: {top_100_scores[i]}, Label: {top_100_labels[i]}")

# Визуализация
# Вычисляем разумные границы для оси X
x_min = np.min(all_scores)
x_max = np.max(all_scores)
# Добавляем 10% от диапазона с каждой стороны для визуального комфорта
padding = (x_max - x_min) * 0.1
x_lim = (x_min - padding, x_max + padding)

plt.subplot(1, 2, 1)
n_normal = sum(filtered_metadata == 1)
n_anomaly = sum(filtered_metadata != 1)

# Гистограмма с автоматическим подбором bin edges в пределах x_lim
plt.hist(all_scores[filtered_metadata == 1], 
         bins=np.linspace(x_lim[0], x_lim[1], 50),
         alpha=0.5, label=f'Normal (n={n_normal})', color='blue')
plt.hist(all_scores[filtered_metadata != 1], 
         bins=np.linspace(x_lim[0], x_lim[1], 50),
         alpha=0.5, label=f'Anomaly (n={n_anomaly})', color='red')

# Настройка осей и оформления
plt.xlim(x_lim)  # Фиксируем разумные границы
plt.xlabel('Anomaly Score', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Distribution of Anomaly Scores', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.2)

# Добавляем линию порога только если она попадает в диапазон
threshold = np.percentile(all_scores, 5)
if x_lim[0] <= threshold <= x_lim[1]:
    plt.axvline(threshold, color='black', linestyle='--', linewidth=1,
                label=f'5% Threshold ({threshold:.2f})')
    plt.legend(fontsize=9)

# Оптимизируем использование пространства
plt.tight_layout()

# 2. Пространство признаков с аномалиями
plt.subplot(1, 2, 2)
plt.scatter(filtered_data[:, 0], filtered_data[:, 1], 
           c=np.where(filtered_metadata == 1, 'blue', 'red'), 
           alpha=0.3, s=5, label='Normal')
plt.scatter(filtered_data[top_100_indices, 0], filtered_data[top_100_indices, 1], 
           c='yellow', edgecolor='k', s=50, label='Top 100 Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Feature Space with Anomalies')

plt.tight_layout()
plt.savefig('isolation_forest_results.pdf', format='pdf', bbox_inches='tight')