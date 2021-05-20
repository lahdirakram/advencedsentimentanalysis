from flask import Flask, request, render_template
from NLP import Sentiment
from numpy import argmax

app = Flask(__name__)
sentiment_analyser = Sentiment('cpu',512)
labels = ['positif','negatif']



def edit_response(response):
    if "charset=utf-8" not in response.content_type:
        response.content_type = response.content_type.split(';')[0]+"; charset=utf-8"
    return response

app.after_request(edit_response)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/predict")
def predict():
    text = request.args.get('text')
    prediction = sentiment_analyser.predict(text)
    return {
        'probabilite': {
            'positif' : prediction[0],
            'negatif' : prediction[1]
            },
        'sentiment': labels[argmax(prediction)],
        'text' : text
        }