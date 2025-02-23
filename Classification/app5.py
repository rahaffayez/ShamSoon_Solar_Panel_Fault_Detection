from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite
from pyngrok import ngrok  # Import ngrok

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="my_solarmodel.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# Predict function
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels[predicted_class_index]
    confidence = float(output_data[0][predicted_class_index])

    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_fault():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))  # Fixed: Added missing closing parenthesis

        # Predict the class
        predicted_class, confidence = predict(image)

        # Return the result
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Set the ngrok auth token here
    ngrok.set_auth_token('2sV7t8ID8Ezdy2IJEB3yngswQf7_81vvYkxHYuNe7z6AYBC2A')  
    
    # Open a tunnel to the Flask app (running on port 5000)
    public_url = ngrok.connect(5000)
    print(f"Flask app is live at {public_url}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
