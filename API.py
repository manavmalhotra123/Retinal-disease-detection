from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('trained_model.h5')

# Define class labels
class_labels = ['normal', 'cataract', 'glaucoma', 'retina disease']  # Adjust with your class labels

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the image
    image = request.files['image']
    image_data = image.read()
    image_stream = io.BytesIO(image_data)
    image_array = tf.keras.preprocessing.image.load_img(image_stream, target_size=(224, 224))
    image_array = img_to_array(image_array)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    # Prepare response
    response = {
        'predicted_class': predicted_class_label,
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
