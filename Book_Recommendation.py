import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained Keras model
model = load_model("model.h5py")

# Function to preprocess input data
def preprocess_input(input_data, label_encoder_dict=None, scaler=None):
    if label_encoder_dict is None:
        label_encoder_dict = {}
        scaler = StandardScaler()

    columns = ['User-ID', 'ISBN']
    input_data = pd.DataFrame(input_data, columns=columns)

    for column in input_data.select_dtypes(include=['object']).columns:
        if column not in label_encoder_dict:
            label_encoder_dict[column] = LabelEncoder()
            input_data[column] = label_encoder_dict[column].fit_transform(input_data[column])

    numerical = ['User-ID', 'ISBN']
    input_data[numerical] = scaler.fit_transform(input_data[numerical])
    return input_data

# Function to make predictions
def make_prediction(input_data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)

    # Ensure that input_data has the correct data type
    preprocessed_data = preprocessed_data.astype(np.int64)

    # Make predictions using the loaded Keras model
    predictions = model.predict(preprocessed_data)

    return predictions

# Streamlit app
def main():
    st.title("Book Recommendation Web App📚")

    # Create input widgets (example: numeric input fields)
    user_id = st.number_input("Enter User-ID:", min_value=0.0, value=None)
    isbn = st.number_input("Enter ISBN:", min_value=0, value=None)

    # # Get user input
    # user_id_input = st.text_input("Enter User ID:")
    # isbn_input = st.text_input("Enter ISBN:")

    # Make predictions
    if st.button('Get Recommendation rating'):
        input_data = np.array([[user_id, isbn]])
        prediction = make_prediction(input_data)

        # Display the prediction
        st.success(f"Your recommendation rating is: {prediction[0][0]:.4f}")

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
        

if __name__ == "__main__":
    main()