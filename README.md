## UMAP Visualization of Iris Data

The UMAP plot below shows clusters formed by the three species of iris flowers:

![UMAP Visualization](umap_visualization.png)

You can regenerate this plot by running the `visualize.py` script:

```bash
python visualize.py

### 2. **Modifying GitHub Action to Produce Model as an Artifact**

To produce a trained model as an artifact, you need to modify the GitHub Actions CI configuration file (`ci.yml`), train the model in `train.py`, and pickle the result for later use.

Here’s an outline of how to do this:

1. **Ensure that `train.py` saves the trained model**:

```python
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'model.pkl')
