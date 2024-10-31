
# Import required libraries
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Set up FastAPI
app = FastAPI()

# Load data from CSV
data = pd.read_csv("sample_training_data.csv")

# Preprocess data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)

# Define collaborative filtering model
num_users = data['user_id'].nunique()
num_items = data['item_id'].nunique()
embedding_dim = 50

user_input = tf.keras.layers.Input(shape=(1,))
item_input = tf.keras.layers.Input(shape=(1,))

user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
item_embedding = tf.keras.layers.Embedding(input_dim=num_items, output_dim=embedding_dim)(item_input)

dot_product = tf.keras.layers.Dot(axes=1)([user_embedding, item_embedding])
output = tf.keras.layers.Flatten()(dot_product)

model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
user_train = train_data['user_id'].values
item_train = train_data['item_id'].values
ratings_train = train_data['rating'].values

model.fit([user_train, item_train], ratings_train, epochs=5, batch_size=32)

# Building similarity scoring functions
item_embeddings = model.get_layer('embedding').get_weights()[0]

def get_similar_items(item_id, top_k=5):
    item_vector = item_embeddings[item_id]
    similarities = np.dot(item_embeddings, item_vector)
    return np.argsort(similarities)[-top_k:]

def get_user_recommendations(user_id, top_k=5):
    user_vector = model.get_layer('embedding').get_weights()[0][user_id]
    scores = np.dot(item_embeddings, user_vector)
    return np.argsort(scores)[-top_k:]

# Set up FastAPI endpoints
@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    recommendations = get_user_recommendations(user_id)
    return {"user_id": user_id, "recommendations": recommendations.tolist()}

@app.get("/similar_items/{item_id}")
async def similar_items(item_id: int):
    similar = get_similar_items(item_id)
    return {"item_id": item_id, "similar_items": similar.tolist()}

# Model evaluation
ratings_pred = model.predict([test_data['user_id'], test_data['item_id']])
mae = mean_absolute_error(test_data['rating'], ratings_pred)
print(f"Mean Absolute Error: {mae}")

# Running FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
