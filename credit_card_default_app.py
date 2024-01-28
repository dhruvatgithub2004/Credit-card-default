import pickle
import streamlit as st
with open('Scaling.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)


with open('PCA_fully_applied.pkl', 'rb') as file:
    loaded_pca = pickle.load(file)

with open('credit_card_default.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('Encoding.pkl','rb') as file:
    loaded_encoding = pickle.load(file)


# Streamlit app
st.title('Credit Card Default Prediction App')

uploaded_file = st.file_uploader('Upload a CSV file for prediction:', type=['csv'])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)

    st.write("Original Data:")
    st.write(data.head())

    data["CAT_GAMBLING_ENCODED"] = loaded_encoding.transform(data[['CAT_GAMBLING']])
    data = data.drop("CAT_GAMBLING", axis=1)

    data_scaled = loaded_scaler.transform(data)

    data_pca = loaded_pca.transform(data_scaled)[:, :40]

    predictions = loaded_model.predict(data_pca)

    data['Predictions'] = predictions
    st.write('Predictions:')
    st.write(data.head())

    st.download_button('Download Predictions', data.to_csv(index=False), file_name='predictions.csv', key='download_button')
