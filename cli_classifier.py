import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Model Training (as done in your original Spam.py) ---
# This part trains the model every time the script runs.
# In a production environment, you would save and load the trained model and vectorizer.

try:
    # Load and prepare the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Normalize text to lowercase
    df['text'] = df['text'].str.lower()

    # Split the data (70% training, 30% testing)
    # Note: X_test, y_test are not used for classification here, but needed for training setup
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42
    )

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    # X_test_vec = vectorizer.transform(X_test) # Not needed for training, but would be for evaluation

    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("Model trained successfully. Ready for classification.")

except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure the file is in the same directory.")
    print("Cannot proceed with classification without the dataset.")
    exit()
except Exception as e:
    print(f"An error occurred during model training: {e}")
    exit()

# --- User Input and Classification ---
def classify_message(message_text):
    """
    Classifies a single message using the trained model.
    """
    # Normalize the input message (lowercase, consistent with training)
    normalized_message = message_text.lower()

    # Vectorize the new message
    # Use the *same* vectorizer that was fit on the training data
    message_vec = vectorizer.transform([normalized_message])

    # Predict the label
    prediction = model.predict(message_vec)

    return prediction[0] # prediction is an array, take the first element

if __name__ == "__main__":
    print("\n--- SMS Spam Classifier (CLI) ---")
    print("Type a message and press Enter to classify it. Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter message: ")
        if user_input.lower() == 'exit':
            print("Exiting classifier. Goodbye!")
            break

        if user_input.strip() == '':
            print("Please enter some text to classify.")
            continue

        classification_result = classify_message(user_input)
        print(f"Classification: {classification_result.upper()}")
        if classification_result == 'spam':
            print("🚨 This message is likely SPAM! 🚨")
        else:
            print("✅ This message is likely HAM (not spam). ✅")