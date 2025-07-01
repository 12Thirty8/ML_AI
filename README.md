SMS Spam Classifier
This is a simple web application built with Streamlit that classifies SMS messages as either "ham" (not spam) or "spam" using a Multinomial Naive Bayes model.

Features
User-friendly Interface: Easily input text messages through a web interface.

Real-time Classification: Get instant predictions on whether a message is spam or ham.

Machine Learning Powered: Utilizes a pre-trained (on application start) Multinomial Naive Bayes model for text classification.

Setup Instructions
To run this application locally, follow these steps:

Clone the repository (if applicable) or create the necessary files:
Ensure you have the following files in the same directory:

streamlit_classifier.py (the Python code for the Streamlit app)

spam.csv (the dataset used for training the model)

requirements.txt (listing the Python dependencies)

Create requirements.txt:
Make sure your requirements.txt file contains the following lines:

pandas
scikit-learn
streamlit

Install Dependencies:
Open your terminal or command prompt, navigate to the directory where you saved the files, and install the required Python libraries using pip:

pip install -r requirements.txt

Run the Application:
Once the dependencies are installed, you can start the Streamlit application by running the following command in your terminal:

streamlit run streamlit_classifier.py

This command will open the application in your default web browser (usually at http://localhost:8501).

Usage
Open the application in your web browser.

Type or paste an SMS message into the text area provided.

Click the "Classify Message" button.

The application will display whether the message is classified as "SPAM" or "HAM".

Screenshot
![Screenshot of Streamlit App](assets/UI.png)
