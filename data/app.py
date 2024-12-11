from flask import Flask, jsonify, request, render_template
from joblib import load
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Keras model
model = tf.keras.models.load_model("amazo.h5")

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
        
        if not review.strip():
            return jsonify({'prediction_text': 'Please enter a review.'})
        
        # Load the vectorizer
        vectorizer = load('amazo (1).save')

        # Debug: Check the vocabulary size
        print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")  # Should be 3000

        # Transform the review and ensure it has the correct shape (1, 3000)
        review_vectorized = vectorizer.transform([review]).toarray()
        print(f"Shape of the review vector: {review_vectorized.shape}")

        # Ensure the vector has the expected shape (1, 3000)
        if review_vectorized.shape[1] != 3000:
            raise ValueError(f"Expected input vector of shape (1, 3000), but got {review_vectorized.shape}")

        # Make a prediction with the model
        prediction = model.predict(review_vectorized)
        
        # Classify the result as Positive or Negative based on the model output
        prediction_text = "Positive review" if prediction[0] > 0.5 else "Negative review"

        return jsonify({'prediction_text': prediction_text})
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'prediction_text': f'Error during prediction: {str(e)}'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
