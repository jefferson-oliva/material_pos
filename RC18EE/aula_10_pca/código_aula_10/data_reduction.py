'''
Data reduction methods

Author: Jefferson Tales Oliva
'''


'''
This method is a PCA (principal component analysis) implementation.

- given matrix A with dimension m (instances) x n (features) and d value, which is the new amount of features. PCA aims to reduce a matrix Amxn to Amxd

- In this method version, the z-score technique was not implemented


- PCA is applied in following steps:

    1 - feature standardization (PCA is sensible to the measure scale): A = StandardScaler().fit_transform(A)

    2 - Measure the average for each line of the matrix A: u = A.mean(0)

    3 - Measuring covariance: C = (1 / (m - 1)) * (A - u).T @ (A - u)

    4 - Get eigenvalues and eigenvectors: eig_vals, eig_vecs = numpy.linalg.eig(C)

    5 - Get eigenvector subset: W = eig_vecs[:, 0 : d]

    6 - PCA result: Y = A @ W # Y = A W


- apply_PCA parameters are presented bellow:

    - table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

    - features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_PCA(df, features, cl, 2)
'''
def apply_PCA(table, features, class_label, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    # standardize features: PCA is sensible to the measure scale
    x = StandardScaler().fit_transform(x)

    # apply PCA
    pca = PCA(n_components = n_components)
    pc = pca.fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Factor analysis implementation

Factor analysis is applied as follows:
    - A - A.mean(0) = L @ F + error

    - A is a matrix input with dimension m x n

    - L is an m x k (amount of components) loading matrix

    - F is a k x n common factor matrix

    - error[i] are unobserved stochastic error terms

    - E(F) = 0

    - cov(F) = I

    - cov(A - A.mean(0)) = cov(L @ F + error) = L @ cov(F) @ L.T + cov(error) = L @ L.T + cov(error)

    - A = L @ F + A.mean(0) + error

    - If F[i] is given: p(A[i] | F[i]) = N(L @ F + A.mean(0), cov(error))

    - For a complete probabilistic model we also need a prior distribution for the latent variable F: p(A) = N(A.mean(0), L @ L.T + cov(error))

    - u = A.mean(0)

    - cov(A - u) = L @ L.T + cov(error)

    - C = numpy.cov(A.T) ou C = (1 / (m - 1)) * (A - u).T @ (A - u)

    - p(A) = N(u, C) # multivariate normal distribution

    -

    -

    - F_scores = (A - u).T @ F (será)

    -

    -

    -

    -

    -

    - # numpy.exp(-0.5 * (A - u).T @ numpy.linalg.inv(C) @ (A - u)) / numpy.sqrt(numpy.power(2 * math.pi, len(u)) * numpy.linalg.det(C))

    - u = A.mean(0)

    - Z = (A - A.mean(0)) / A.std(0)

    - C = (1 / (m - 1)) * (Z).T @ (Z)

    - eig_vals, eig_vecs = numpy.linalg.eig(C)

    - L = numpy.diag(eig_vals)

    - loadings = eig_vecs @ numpy.sqrt(L)

    - B = eig_vecs @ numpy.diag(numpy.power(eig_vals, -0.5)) # score matrix

    - F = Z @ B# factor scores

    -

    -

    -


Parameters:

    - table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

    - features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_factor_analysis(df, features, cl, 2)
'''
def apply_factor_analysis(table, features, class_label, n_components):
    from sklearn.decomposition import FactorAnalysis
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    pc = FactorAnalysis(n_components = n_components, random_state = 0).fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Singular value decomposition


- SVD is applied as follows:

    - Y = USV^T. For data reduction, the trasformation only must be US

    - U = eig(x @ x.T) #U is a m x m orthonormal matrix of 'left-singular' (eigen)vectors of  xx^T

    - lmbV, _ = eig(x.T @ x) #V is a n x n orthonormal matrix of 'right-singular' (eigen)vectors of  x^T

    - S = sqrt(diag(abs(lmbV))[:n_components,:]) # S is a m x n diagonal matrix of the square root of nonzero eigenvalues of U or V


- apply_SVD parameters:

    - table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

    - features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_SVD(df, features, cl, 2)
'''
def apply_SVD(table, features, class_label, n_components):
    from numpy import diag
    from scipy.sparse.linalg import svds
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, class_label)

    # apply SVD
    u, s, _ = svds(x, n_components)
    pc = u @ diag(s) # If we use V^T in this operation, the pc will have the original dimension

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Sparse random projections

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_SRP(df, features, cl, 2)
'''
def apply_SRP(table, features, class_label, n_components):
    from sklearn.random_projection import SparseRandomProjection
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, class_label)

    rp = SparseRandomProjection(n_components = n_components, dense_output = True, random_state = 420)
    pc = rp.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Gaussian random projections

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_GRP(df, features, cl, 2)
'''
def apply_GRP(table, features, class_label, n_components):
    from sklearn.random_projection import GaussianRandomProjection
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, class_label)

    rp = GaussianRandomProjection(n_components = n_components,  eps = 0.1, random_state = 420)
    pc = rp.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Feature agglomeration

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_feature_agglomeration(df, features, cl, 2)
'''
def apply_feature_agglomeration(table, features, class_label, n_components):
    from sklearn.cluster import FeatureAgglomeration
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, class_label)

    fa = FeatureAgglomeration(n_clusters = n_components, linkage = 'ward')
    pc = fa.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Independent component analysis

- table (DataFrame): it is a feature dataset, where the lines are instances and columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class (e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_ICA(df, features, cl, 2)
'''
def apply_ICA(table, features, class_label, n_components):
    from sklearn.decomposition import FastICA
    from AutoGenMR import feature_file_processor
    #from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, class_label)

    ica = FastICA(n_components = n_components, random_state = 420)
    pc = ica.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[class_label]])
