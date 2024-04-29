import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def predict_category(user_interest):
    module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
    model = hub.load(module_url)

    #trained model(from testValidate.py)
    trained_model = tf.keras.models.load_model("C:/Users/testmodel.keras")

    new_data = pd.DataFrame({"Learning mode": [user_interest]})

    #embed to number vector
    new_embeddings = model(new_data["Learning mode"].tolist()).numpy()

    #number vectors put into keras.predict
    predictions = trained_model.predict(new_embeddings)

    #convert to class labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("C:/Users/testLabelEncoder.npy", allow_pickle=True)
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    for text, predicted_class in zip(new_data["Learning mode"], predicted_classes):
        print(f"Learning mode: {text}, Predicted learning tool: {predicted_class}")
        print('predicted_class: ',predicted_class)
        return predicted_class
        
learning_mode = input("Key in your learning mode：")

print(predict_category(learning_mode))



