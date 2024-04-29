from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import tensorflow as tf
from flask_cors import CORS
import keras
import subprocess
from recommendation import recommend
from predictCategory import predict_category

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')
#keras 3.1.1 tensorflow 2.16.1
@app.route("/")
def home():
    return submit_questions()

@app.route("/submit_questions", methods=["GET"])
def submit_questions():
    if request.method == "GET":
        interest = request.args.get("interest")
        learningMode = request.args.get("learningMode")
        print('interest: ',interest)
        print("learningMode",learningMode)
        predictedCategory = predict_category(learningMode)
        print("predicted: ",predictedCategory)
        recommendations = recommend(interest, learningMode)
        recommendations = recommendations.split("\n")
        print("Recommendation output: ",recommendations)
        return recommendations
    else:
        return "Method Not Allowed: Only GET requests are accepted.", 405

if __name__ == "__main__":    
    app.run()