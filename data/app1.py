from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

# Example training data (replace with your actual training data)
reviews = [
    "This is an amazing product, with great features.",
    "Very bad product, doesn't work as expected.",
    "I am really satisfied with the quality and performance."
    # Add more reviews here...
]

# Train the vectorizer with max_features=3000
vectorizer = CountVectorizer(max_features=3000)
vectorizer.fit(reviews)

# Save the vectorizer
dump(vectorizer, 'amazo.save')
print("Vectorizer saved successfully with 3000 features.")
