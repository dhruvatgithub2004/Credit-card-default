import os
import pickle
import pandas as pd
import streamlit as st

# Get the absolute path to the directory containing the Streamlit app script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the full paths to the pickled files
scaling_path = os.path.join(script_directory, 'Scaling.pkl')
pca_path = os.path.join(script_directory, 'PCA_fully_applied.pkl')
model_path = os.path.join(script_directory, 'credit_card_default.pkl')
encoding_path = os.path.join(script_directory, 'Encoding.pkl')

# Log paths to a file
log_file_path = os.path.join(script_directory, 'log.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Current Working Directory: {os.getcwd()}\n")
    log_file.write(f"Script Directory: {script_directory}\n")
    log_file.write(f"Scaling Path: {scaling_path}\n")
    log_file.write(f"PCA Path: {pca_path}\n")
    log_file.write(f"Model Path: {model_path}\n")
    log_file.write(f"Encoding Path: {encoding_path}\n")

# Load the pickled files
with open(scaling_path, 'rb') as file:
    loaded_scaler = pickle.load(file)

with open(pca_path, 'rb') as file:
    loaded_pca = pickle.load(file)

with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

with open(encoding_path, 'rb') as file:
    loaded_encoding = pickle.load(file)

# Streamlit app
st.title('Credit Card Default Prediction App')

uploaded_file = st.file_uploader('Upload a CSV file for prediction:', type=['csv'])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    st.write("Original Data:")
    st.write(data.head())

    # Apply encoding consistently with training
    data["CAT_GAMBLING_ENCODED"] = loaded_encoding.transform(data[['CAT_GAMBLING']])
    data = data.drop("CAT_GAMBLING", axis=1)

    # Scale the data using the same scaler used during training
    data_scaled = loaded_scaler.transform(data)

    data_pca = loaded_pca.transform(data_scaled)[:, :40]

    predictions = loaded_model.predict(data_pca)

    # Add predictions to the original dataframe
    data['Predictions'] = predictions

    # Display the first few rows of the dataframe
    st.write('Predictions:')
    st.write(data.head())

    # Allow the user to download the predictions as a CSV file
    st.download_button('Download Predictions', data.to_csv(index=False), file_name='predictions.csv', key='download_button')
