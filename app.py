import streamlit as st
import requests

# Set up the Streamlit app
st.title("Retinal Scanner")

# API endpoint URL
url = 'http://127.0.0.1:5000/predict'  # Replace with your API endpoint URL

# Display file uploader for image selection
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Make a prediction when the user uploads an image
if image_file is not None:
    # Send POST request with the image file
    files = {'image': image_file}
    response = requests.post(url, files=files)

    # Check response status code
    if response.status_code == 200:
        try:
            # Get the response data
            data = response.json()

            # Display the predicted class label and probabilities
            st.subheader('Prediction Result')
            st.write('Predicted Class:', data['predicted_class'])
            
        except ValueError as e:
            st.write('Error: Invalid JSON response')
    else:
        st.write('Error:', response.status_code)
