# Task 6: K-Nearest Neighbors (KNN) Classification with CSV Dataset + Decision Boundaries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load dataset (👉 Change path to where you save iris.csv)
file_path = r"C:\Users\Admin\Downloads\task 6\Iris.csv"   # <<-- put your dataset path here
df = pd.read_csv(file_path)

# 2. Select only first 2 features for visualization
X = df.iloc[:, [0, 1]].values   # sepal_length, sepal_width
y = df.iloc[:, -1].values       # species

# 3. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Experiment with different values of K
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"K={k} -> Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# 6. Confusion Matrix for K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (K=5)")
plt.show()

# 7. Decision Boundary Visualization (using first 2 features)
h = 0.02  # step size for mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
plt.scatter(X[:, 0], X[:, 1], c=pd.factorize(y)[0], edgecolor="k", cmap=plt.cm.Set1)
plt.title("KNN Decision Boundaries (K=5)")
plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.show()