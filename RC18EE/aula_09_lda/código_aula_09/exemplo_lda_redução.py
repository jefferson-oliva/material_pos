import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Gerar um conjunto de dados sintético com 3 classes e 5 features
X, y = make_classification(n_samples=150, n_features=5, n_informative=3,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           class_sep=2.0, random_state=42)

# Aplicar LDA para reduzir para 2 dimensões
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plotar os dados projetados
colors = ['r', 'g', 'b']
markers = ['o', 's', '^']
labels = ['Classe 0', 'Classe 1', 'Classe 2']

plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1],
                color=colors[i], marker=markers[i], label=labels[i], alpha=0.7)
    
plt.title('Redução de Dimensionalidade com LDA')
plt.xlabel('Componente LD1')
plt.ylabel('Componente LD2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#----------------------------------------------------------
#Geração da matriz W
import numpy as np
from numpy.linalg import inv, eig

# X: matriz de dados (n x d), y: rótulos (n,)
def calcular_matriz_W(X, y, k):
    classes = np.unique(y)
    n_features = X.shape[1]
    mean_global = np.mean(X, axis=0)

    # Inicializar matrizes
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    for cls in classes:
        X_c = X[y == cls]
        mean_c = np.mean(X_c, axis=0)
        # Intra-classe
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
        # Entre-classes
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_global).reshape(n_features, 1)
        S_B += n_c * (mean_diff).dot(mean_diff.T)

    # Resolver S_W^-1 * S_B
    eig_vals, eig_vecs = eig(inv(S_W).dot(S_B))

    # Ordenar pelos maiores autovalores
    idx = np.argsort(np.abs(eig_vals))[::-1]
    eig_vecs = eig_vecs[:, idx]
    
    # Selecionar os k autovetores principais
    W = eig_vecs[:, :k]
    return W
    
    
W = calcular_matriz_W(X, y, k=2)
X_proj = X @ W  # Dados projetados
