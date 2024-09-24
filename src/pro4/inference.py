from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import joblib

# Create the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('model.pkl')

# Define a class for the input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create an endpoint for predictions
@app.post("/predict/")
def predict(iris: IrisInput):
    # Convert input data to a NumPy array
    input_data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    
    # Make the prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Return the prediction
    return {"prediction": int(prediction[0])}
