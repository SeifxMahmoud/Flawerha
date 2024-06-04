from PIL import Image
import numpy as np
from flask import Flask, render_template, request
import pickle
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)

model = pickle.load(open('C:\\Users\\sief x\\Flowerha\\Models\\model.pk1', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
        file = request.files['file']
        filename = secure_filename(file.filename)
        image = Image.open(file)
        image = image.resize((224, 224))  # Resize image to match model input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        predictions = model.predict(image)
        output = np.argmax(predictions, axis=1)[0]  # Get the numerical value of the predicted class (0-4)
        return render_template('index.html', prediction_text='Predicted class is {}'.format(output))

if __name__ == '__main__':
        app.run(debug=True)

