import numpy as np
from flask import Flask, request, render_template
import pickle
import nbimporter
from pipeline import clean_text

application = Flask(__name__) 


model = pickle.load(open('model.pkl', 'rb'))

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    text = request.form['text']

    cleaned_text = clean_text(text) # preprocess text

    final_text = np.array([cleaned_text])  

    
    prediction_news = model.predict(final_text)

    # Convert numerical label back to category using the loaded LabelEncoder
    output = label_encoder.inverse_transform(prediction_news)[0]  

    return render_template('result.html', prediction_text=f'News Category: {output}')

if __name__ == "__main__":
    application.run(debug=True)
