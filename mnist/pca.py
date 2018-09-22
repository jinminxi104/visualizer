from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

digits = datasets.load_digits()

pca = PCA()
X_pca = pca.fit_transform(digits.data)

plt.figure(figsize=(4,4))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.axis('equal');
plt.show()
