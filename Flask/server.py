# Aplicación Flask que use tran_model.py
# Path: Flask/server.py

from flask import Flask, request, jsonify
from load_model import predict

Flask = Flask(__name__)

@Flask.route('/')
def index():
    return 'Flask server is running'

@Flask.route('/predict', methods=['GET'])
def get_prediction():
    answer_text = request.args.get('answer_text')
    source_text = request.args.get('source_text')

    prediction = predict(answer_text, source_text)
    return prediction
