import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle

"""
- like a simple web framework 
- basic idea of flask infrastucture is you can make routes which is like pages 
on a website when you send a response with the URL, you can respond by 
sending back a html page.  Sending a lot of requests and responding with webpage
"""

def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
    # stub input features
    request_json = request.get_json()
    x = request_json['input']
    #print(x)
    x_in = np.array(x).reshape(1,-1)
    # x = np.array(data_in).reshape(1,-1)
    # load model
    model = load_models()
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)
