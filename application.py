import numpy as np
from flask import Flask, request, render_template
import pickle

application = Flask(__name__) 

# Load the trained model and label encoder
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
    # Get input text from the form
    text = request.form['text']

    # Convert input text into a NumPy array (modify if preprocessing is needed)
    final_features = np.array([text])  

    # Predict the category (assuming the model works with raw text)
    prediction = model.predict(final_features)

    # Convert numerical label back to category using the loaded LabelEncoder
    output = label_encoder.inverse_transform(prediction)[0]  

    return render_template('result.html', prediction_text=f'News Category: {output}')

if __name__ == "__main__":
    application.run(debug=True)
