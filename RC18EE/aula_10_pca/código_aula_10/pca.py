import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Passo 1: Carregar e padronizar os dados
iris = load_iris()
X = iris.data
y = iris.target

# Padronização: média 0, desvio padrão 1
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Passo 2: Calcular a matriz de covariância
cov_matrix = np.cov(X_std.T)  # Transposta para variáveis nas colunas

# Passo 3: Calcular autovalores e autovetores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Passo 4: Ordenar autovalores e autovetores
idx = np.argsort(eigenvalues)[::-1]  # Ordem decrescente
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Passo 5: Selecionar os k principais componentes
k = 2
principal_components = eigenvectors[:, :k]

# Passo 6: Projetar os dados
X_pca = X_std @ principal_components  # Produto matricial

# Passo extra: Variância explicada
explained_variance = eigenvalues / np.sum(eigenvalues)
print(f"Variância explicada: {explained_variance[:k]}")
print(f"Variância explicada acumulada: {np.cumsum(explained_variance)[:k]}")

# Passo 7: Visualizar os dados projetados
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA do zero - Iris dataset')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(True)
plt.show()

