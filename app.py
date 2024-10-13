from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = load_model('model_1.h5')

# Initialize MediaPipe Hands for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Label mapping for the output classes
label_mapping_reverse = {i: label for i, label in enumerate([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 
    'Y', 'Z', ''])}

def normalize_landmarks(landmarks):
    base_x, base_y = landmarks[0]  # Wrist is the base point
    normalized_landmarks = [(x - base_x, y - base_y) for x, y in landmarks]
    return normalized_landmarks

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get the image data from the request
    img_data = data['image']
    
    # Decode the image from base64
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    results_hand = hands.process(frame_rgb)

    predicted_label = ''
    
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            # Extract and normalize the landmarks
            landmarks = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
            normalized_landmarks = normalize_landmarks(landmarks)

            # Convert the normalized landmarks to a numpy array
            if normalized_landmarks:
                normalized_landmarks = np.array(normalized_landmarks).reshape(1, 21, 2)

                # Make prediction using the model
                prediction = model.predict(normalized_landmarks)
                predicted_label_index = np.argmax(prediction)
                predicted_label = label_mapping_reverse.get(predicted_label_index, '')

    return jsonify({'predicted_label': predicted_label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
