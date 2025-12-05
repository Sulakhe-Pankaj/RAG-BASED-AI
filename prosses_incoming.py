import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1:1.5b",
        "model": "llama3.2",
        "prompt": prompt,
        "stream":False 
    })
    
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')


incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0] 

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx] 
# print(new_df[["title", "number", "text"]])

prompt = f'''
i am teaching you html and css using apana college course. Here videos subtitle chunks are given.
The user will ask question related to html and css. You have to answer the question based on the chunks only.
Here are the chunks: {new_df[["title", "number","text"]].to_json(orient="records")}
-----------------------------------------------------------------------------------------------
"{incoming_query}"
User asked this Question related to html and css. Provide a precise answer based on the chunks' Where and how much information is given in the chunks.
and guide the user to the chunk number for better understanding. If users ask unrelated question to html and css then
respond with "I am sorry, I can only answer questions related to html and css as I have been trained on apna college html and css course."
'''

with open("prompt.txt", "w") as f: 
    f.write(prompt)

response = inference(prompt)['response']
print(response)
with open("response.txt", "w") as f:
    f.write(response)

print(inference(prompt))

# for index, item in new_df.iterrows():
    # print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

