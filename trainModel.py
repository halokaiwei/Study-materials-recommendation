import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
model = hub.load(module_url)
print("module %s loaded" % module_url)
def embed(input):
    return model(input)
df = pd.read_csv("C:/Users/FYP/trainData.csv")
print(df.head())
df = df[["Learning mode", "Learning tool"]]  
df = df.dropna()
df = df.reset_index()
#convert to number
label_encoder = LabelEncoder()
df['Learning tool'] = label_encoder.fit_transform(df['Learning tool'])

embeddings = embed(df['Learning mode'].tolist()).numpy()

train_embeddings, val_embeddings, train_labels, val_labels =train_test_split(embeddings, df['Learning tool'], test_size=0.2, random_state=42)

#model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512,)),  #US.Encoder 512 shape
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(df['Learning tool'].unique()), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model
history = model.fit(train_embeddings, train_labels,
                    epochs=10,
                    batch_size=512,
                    validation_data=(val_embeddings, val_labels),
                    verbose=1)

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("C:/Users/testmodel.keras")
np.save("C:/Users/testLabelEncoder.npy", label_encoder.classes_)

