import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

df = pd.read_csv("C:/Users/FYP/Tutorial_List.csv")
print(df.head())
df = df[["Topic", "Learning_tool", "Keywords", "Description"]]
df = df.dropna()
df = df.reset_index()
keywords = list(df['Keywords'])
embeddings = embed(keywords)
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(11,6))
plt.title("Embedding Space")
plt.scatter(emb_2d[:,0],emb_2d[:,1])
plt.show()

nn = NearestNeighbors(n_neighbors=10)
nn.fit(embeddings)

def recommend(text, Learning_tool):
    threshold = 0.2
    emb = embed([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    
    
    input_emb = tf.reshape(emb, [1, -1])[0].numpy()  #change to vector

    Learning_tool_neighbors = [neighbor for neighbor in neighbors if df.loc[neighbor, 'Learning_tool'] == Learning_tool]

    selected_recommendations = []
    for neighbor_idx in Learning_tool_neighbors:
        neighbor_keywords = keywords[neighbor_idx].split(', ')
        neighbor_emb = embed(neighbor_keywords)[0].numpy()  #embed keywords
        similarities = cosine_similarity([input_emb], [neighbor_emb])[0]

        #check the similiarity > threshold
        if any(similarity >= threshold for similarity in similarities):
            topic = df.loc[neighbor_idx, 'Topic']
            description = df.loc[neighbor_idx, 'Description']
            recommendation = f"| {topic} | {description} |"
            selected_recommendations.append(recommendation)

    print("Learning tool neighbors count:", len(selected_recommendations))

    #5 recommend
    if len(selected_recommendations) < 5:
        remaining_neighbors = [neighbor for neighbor in range(len(df)) if neighbor not in Learning_tool_neighbors]

        for neighbor in remaining_neighbors:
            neighbor_keywords = keywords[neighbor].split(', ')
            similarities = cosine_similarity([input_emb], [embed(neighbor_keywords)[0].numpy()])[0]

            if any(similarity >= threshold for similarity in similarities) and text in keywords[neighbor]:
                topic = df.loc[neighbor, 'Topic']
                description = df.loc[neighbor, 'Description']
                recommendation = f"| {topic} | {description} |"
                selected_recommendations.append(recommendation)

    print("Learning tool neighbors count:", len(selected_recommendations))
    header = "| Topic  | Description |"
    return f"{header}\n" + "\n".join(selected_recommendations)

print(recommend('coding','videos'))
