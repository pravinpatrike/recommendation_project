from fastapi import FastAPI, Query
from recommend import recommend_verses
import pandas as pd

# Load the dataset
DATA_PATH = "../data/raw/temp_dataset.xlsx"
df = pd.read_excel(DATA_PATH)
df = df.fillna('')

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Bhagavad Gita Life Advisor API!"}

@app.get("/recommend")
def get_recommendations(query: str = Query(..., min_length=1)):
    recommendations = recommend_verses(query, df)
    return {"query": query, "recommendations": recommendations}

