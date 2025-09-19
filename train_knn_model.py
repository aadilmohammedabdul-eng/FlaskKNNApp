# train_knn_model.py
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the trained model to a file
joblib.dump(knn, 'knn_model.joblib')

print("Model trained and saved as knn_model.joblib")
