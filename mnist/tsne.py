import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X1 = TSNE(n_components=2).fit_transform(digits.data)

plt.scatter(X1[:,0], X1[:,1])
plt.show()
