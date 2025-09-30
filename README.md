Task 6: K-Nearest Neighbors (KNN) Classification with CSV Dataset

Objective

The objective of this task is to understand and implement the K-Nearest Neighbors (KNN) algorithm for classification problems. This task demonstrates instance-based learning, Euclidean distance calculation, K selection, and decision boundary visualization.


---

Dataset

Dataset: Iris Dataset (CSV format)

Features: Sepal length, Sepal width, Petal length, Petal width

Target: Species (setosa, versicolor, virginica)

Dataset is loaded from a CSV file, allowing flexibility to use any local copy.



---

Tools & Libraries

Python 3

Pandas

NumPy

Matplotlib

Scikit-learn (KNeighborsClassifier, StandardScaler, train_test_split, accuracy_score, confusion_matrix)



---

Steps Implemented

1. Load Dataset

CSV dataset is loaded using Pandas.

Users can set the path to their local iris.csv file.



2. Select Features

Only the first two features (sepal_length, sepal_width) are used for visualization.



3. Normalize Features

Features are standardized using StandardScaler() to improve KNN performance.



4. Train-Test Split

Data is split into training (70%) and testing (30%) sets using train_test_split.



5. Train KNN Classifier

KNN is trained for multiple values of K (1, 3, 5, 7) using KNeighborsClassifier.

Accuracy is calculated for each K value.



6. Evaluate Model

Confusion matrix is generated and visualized for K=5.

Accuracy score is printed for all K values.



7. Decision Boundary Visualization

Decision boundaries are plotted in 2D using the first two features.

Background color indicates predicted class regions.

Scatter points show actual data points colored by species.





---

Outputs

1. Accuracy for different K values:

K=1 -> Accuracy: 1.000
K=3 -> Accuracy: 1.000
K=5 -> Accuracy: 1.000
K=7 -> Accuracy: 1.000

(may vary slightly depending on train/test split)


2. Confusion Matrix (K=5)

Displays correct and incorrect classifications in a grid.


[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]


3. Decision Boundary Plot

Shows KNN classification regions in 2D.

Points represent actual iris samples.

Useful for visualizing how the classifier separates different species.





---

Learning Outcomes

Understanding instance-based learning and how KNN predicts labels based on nearest neighbors.

Learning how feature normalization affects KNN performance.

Observing the effect of different K values on model accuracy.

Visualizing decision boundaries and interpreting model predictions.



---

How to Run

1. Save your Iris dataset as iris.csv.


2. Update the file_path in the code to your CSV location.


3. Run the Python script in any IDE (VS Code, Jupyter Notebook, etc.).


4. Observe the printed accuracy, confusion matrix, and decision boundary# dishap81-task6
Machine Learning project for the Titanic dataset, focusing on predicting passenger survival. Features data exploration, outlier removal, and model training.
