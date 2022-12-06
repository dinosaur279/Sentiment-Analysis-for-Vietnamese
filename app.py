from flask import Flask, render_template, request, url_for
import pickle
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation

vectorizer = TfidfVectorizer()
app = Flask(__name__)

classifier = pickle.load(open('model/model_pipeline.pkl', 'rb'))

def remove_punctuation(text):
    text = ''.join([ch for ch in text if ch not in punctuation])
    return text

@app.route('/')
def home():
	return render_template('home.html')


@app.route("/Predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        my_prediction = classifier.predict([message])
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
