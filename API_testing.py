
import requests

# API endpoint URL
url = 'http://127.0.0.1:5000/predict'  # Replace with your API endpoint URL

# Path to the image file
image_path = 'valid/1_normal/0000_png.rf.f48a3499187d5c0e8bea8165a231add8.jpg'  # Replace with the path to your image file

# Send POST request with the image file
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

# Check response status code
if response.status_code == 200:
    try:
        # Get the response data
        data = response.json()

        # Print the predicted class label and probabilities
        print('Predicted Class:', data['predicted_class'])
        print('Probabilities:', data['probabilities'])
    except ValueError as e:
        print('Error: Invalid JSON response')
else:
    print('Error:', response.status_code)

# Path to the image file
