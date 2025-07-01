import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st # Import Streamlit

# Recommended requirements.txt content for this Streamlit app:
# pandas
# scikit-learn
# streamlit

# --- Model Training (cached for performance) ---
# Use st.cache_resource to ensure the model and vectorizer are trained only once
# when the app starts, even across user interactions.
@st.cache_resource
def train_model_and_vectorizer():
    """
    Loads data, trains the Multinomial Naive Bayes model, and fits the CountVectorizer.
    This function is cached to avoid re-training on every interaction.
    """
    try:
        # Load and prepare the dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']

        # Normalize text to lowercase
        df['text'] = df['text'].str.lower()

        # Split the data (70% training, 30% testing)
        # X_test, y_test are not used for classification here, but needed for training setup
        X_train, _, y_train, _ = train_test_split(
            df['text'], df['label'], test_size=0.3, random_state=42
        )

        # Vectorize the text data
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Train the Multinomial Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        return model, vectorizer

    except FileNotFoundError:
        st.error("Error: 'spam.csv' not found. Please ensure the file is in the same directory as the app.")
        st.stop() # Stop the app if the file is missing
    except Exception as e:
        st.error(f"An error occurred during model training: {e}")
        st.stop() # Stop the app on other training errors

# Train the model and vectorizer when the app starts
model, vectorizer = train_model_and_vectorizer()

# --- Streamlit UI ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️", layout="centered")

st.title("✉️ SMS Spam Classifier")
st.markdown("Enter a message below to classify it as 'ham' (not spam) or 'spam'.")

# Text area for user input
user_input = st.text_area("Enter your message here:", height=150, placeholder="Type your message...")

# Button to trigger classification
if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Normalize the input message (lowercase, consistent with training)
        normalized_message = user_input.lower()

        # Vectorize the new message using the trained vectorizer
        message_vec = vectorizer.transform([normalized_message])

        # Predict the label
        prediction = model.predict(message_vec)[0] # prediction is an array, take the first element

        st.subheader("Classification Result:")
        if prediction == 'spam':
            st.error(f"🚨 This message is likely **{prediction.upper()}**! 🚨")
        else:
            st.success(f"✅ This message is likely **{prediction.upper()}** (not spam). ✅")

st.markdown("---")
st.info("This classifier uses a Multinomial Naive Bayes model trained on the SMS Spam Collection Dataset.")

