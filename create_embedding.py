import pandas as pd
import joblib
import requests

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

df = pd.DataFrame()
df["embedding"] = create_embedding

# Save it safely
joblib.dump(df, "embeddings.joblib")
print("✅ embeddings.joblib created successfully!")
