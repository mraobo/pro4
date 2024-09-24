import plotly.express as px
import umap
import pandas as pd
from sklearn.datasets import load_iris
import os

def generate_umap_plot():
    print("Loading Iris dataset...")
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Applying UMAP for dimensionality reduction...")
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    print("Creating DataFrame...")
    # Create a DataFrame for easy plotting
    df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df['species'] = y

    print("Generating Plotly visualization...")
    # Generate Plotly visualization
    fig = px.scatter(df, x='UMAP1', y='UMAP2', color=df['species'].astype(str),
                     title='UMAP projection of Iris dataset',
                     labels={'species': 'Species'})

    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    print("Saving the plot as 'umap_visualization.svg'...")
    # Save the plot as an image file
    fig.write_image("umap_visualization.svg")

    print("Plot saved successfully!")

if __name__ == "__main__":
    generate_umap_plot()
