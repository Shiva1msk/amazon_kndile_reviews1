from flask import Flask, jsonify, request, render_template
from joblib import load
import tensorflow as tf
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained Keras model
model = tf.keras.models.load_model("amazo (1).h5")

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Get the review from the POST request
        data = request.get_json()
        review = data.get('Sentence', '')
        logging.debug(f"Received review: {review}")
        
        if not review.strip():
            return jsonify({'prediction_text': 'Please enter a review.'})
        
        # Load the vectorizer
        vectorizer = load('amazo (2).save')
        logging.debug(f"Loaded vectorizer with vocabulary size: {len(vectorizer.vocabulary_)}")
        logging.debug(f"Vocabulary contains 'terrible': {'terrible' in vectorizer.vocabulary_}")


        # Transform the review
        review_vectorized = vectorizer.transform([review]).toarray()
        logging.debug(f"Review vector shape: {review_vectorized.shape}")
        logging.debug(f"Vectorized input: {review_vectorized}")
        logging.debug(f"Vocabulary contains 'terrible': {'terrible' in vectorizer.vocabulary_}")


        # Validate vector shape
        if review_vectorized.shape[1] != 3000:
            raise ValueError(f"Expected input vector of shape (1, 3000), but got {review_vectorized.shape}")

        # Make a prediction with the model
        prediction = model.predict(review_vectorized)
        logging.debug(f"Model prediction: {prediction}")
        logging.debug(f"Model confidence score: {prediction[0]}")


        # Interpret the prediction
        prediction_text = "Negative review" if prediction[0] > 0.6 else "positive review"
        return jsonify({'prediction_text': prediction_text})
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'prediction_text': f'Error during prediction: {str(e)}'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
