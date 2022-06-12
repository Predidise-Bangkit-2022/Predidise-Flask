from flask import Flask,request, Response
from flask_restful import Resource, Api
import pandas as pd
from flask_cors import CORS
from keras.models import load_model
import numpy as np
app = Flask(__name__)

CORS(app)

@app.route("/prediction",  methods = ['POST'])
def get():
    userids = request.json['user_id']
    model = load_model('model_new.h5')
    model_input = [np.asarray(userids).astype(np.float32), np.asarray(request.json['tourism_id']).astype(np.float32)]
    prediction = model.predict(model_input)
    predicted_ratings = np.max(prediction, axis=1)
    sorted_index = np.argsort(predicted_ratings)[::-1]
    return Response(pd.Series(sorted_index).to_json(orient='values'), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)