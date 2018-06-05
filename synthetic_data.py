# generate synthetic data for class imbalance illustration

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import pandas as pd 

# Generate the dataset
X_all, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=200, random_state=10)

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)
# Fit and transform x to visualise inside a 2D feature space
X = pca.fit_transform(X_all)
# save original data
X = pd.DataFrame(X)
X.to_csv("Imbalanced Data/X.csv")
pd.DataFrame(y).to_csv("Imbalanced Data/y.csv")



