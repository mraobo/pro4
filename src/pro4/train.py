import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model (RandomForestClassifier in this case)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
joblib.dump(model, 'model.pkl')

print("Model trained and saved as 'model.pkl'")
