# app.py
from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='tf_lite_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check the expected input shape
input_shape = input_details[0]['shape']
print(f"Expected input shape: {input_shape}")

def prepare_image(image):
    # Preprocess the image as required by your model
    image = image.resize((input_shape[2], input_shape[1]))  # Resize to expected dimensions
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.astype(np.float32)  # Ensure the type is FLOAT32
    return np.expand_dims(image_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded image to the static folder
            image_path = os.path.join('static', file.filename)
            image = Image.open(file)
            image.save(image_path)  # Save the image

            processed_image = prepare_image(image)

            # Set the tensor to the input data
            interpreter.set_tensor(input_details[0]['index'], processed_image)

            # Run the model
            interpreter.invoke()

            # Get the output
            prediction = interpreter.get_tensor(output_details[0]['index'])
            result = 'Accident' if prediction[0][0] > 0.5 else 'No Accident'
            return render_template('result.html', result=result, image=file.filename)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)