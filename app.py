from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import logging
logging.basicConfig(level=logging.DEBUG)





# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


@app.route('/')
def home():
    return "Welcome to the Fake News Detection API!"


# Load the trained ML model
model, vectorizer = joblib.load('C:/Users/Subhan IT Solution/Desktop/FakeNewsBackend/fake_news_detection.pkl')

  # Ensure the model file is in the same folder

# Define preprocessing function (optional, based on your model's requirements)
# Preprocessing function
def preprocess_text(text):
    # Use the vectorizer to transform the input text
    return vectorizer.transform([text])  # Ensure this returns the correct sparse matrix format

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request
        app.logger.debug("Request received: %s", request.json)

        # Get the JSON data from the request
        data = request.json
        news_text = data.get('text', '')

        # Validate input
        if not news_text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the input text
        preprocessed_text = preprocess_text(news_text)

        # Make prediction using the model
        prediction = model.predict(preprocessed_text)  # Pass the sparse matrix directly to the model

        # Convert prediction to a Python int
        prediction_result = int(prediction[0])  # Ensure it is JSON serializable

        # Map the prediction to its corresponding label
        label_mapping = {0: "Fake", 1: "Real"}
        prediction_label = label_mapping.get(prediction_result, "Unknown")  # Use "Unknown" as fallback

        # Log and return the result
        app.logger.debug("Prediction label: %s", prediction_label)
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        app.logger.error("Error during prediction: %s", str(e))
        return jsonify({'error': str(e)}), 500





# Run the app
if __name__ == '__main__':
    app.run(debug=True)
