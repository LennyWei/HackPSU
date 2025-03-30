from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import json
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Camera setup
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

GESTURE_FOLDER = 'gestures'
os.makedirs(GESTURE_FOLDER, exist_ok=True)

MODEL_PATH = 'models/gesture_model.h5'
LABEL_MAP_PATH = 'models/label_map.json'

latest_landmarks = []
predicted_label = "Waiting..."

# Load model and label map if available
model = None
label_map = {}
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    model = load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data-collection')
def data_collection():
    return render_template('data-collection.html')

@app.route('/get_landmarks')
def get_landmarks():
    if latest_landmarks:
        return jsonify({"landmarks": latest_landmarks})
    return jsonify({"error": "No landmarks detected"})

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": predicted_label})

@app.route('/collect_data', methods=['POST'])
def collect_data():
    data = request.json
    gesture = data.get('gesture')
    landmarks = data.get('landmarks')

    if not gesture or not landmarks:
        return jsonify({"message": "Invalid gesture or landmarks"}), 400

    gesture_path = os.path.join(GESTURE_FOLDER, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    file_path = os.path.join(gesture_path, f"landmarks_{len(os.listdir(gesture_path))}.json")
    with open(file_path, 'w') as f:
        json.dump(landmarks, f)

    return jsonify({"message": f"Data saved for gesture '{gesture}'"})

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('video_feed.html')

@app.route('/train')
def train():
    return render_template('page.html', title='Train')

@app.route('/test')
def test():
    return render_template('video_feed.html')

@app.route('/contact')
def contact():
    return render_template('page.html', title='Contact')

@app.route('/gestures')
def gestures():
    return render_template('page.html', title='Gestures')

def generate():
    global latest_landmarks, predicted_label
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            # Normalize landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, _ = landmarks[0]
            norm = [(x - base_x, y - base_y) for x, y, _ in landmarks]
            max_val = max([max(abs(x), abs(y)) for x, y in norm])
            if max_val > 0:
                norm = [(x / max_val, y / max_val) for x, y in norm]
            latest_landmarks = norm

            # Predict gesture
            if model is not None and label_map:
                input_data = np.array([np.array(norm).flatten()])
                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_index)]
            else:
                predicted_label = "Hand Detected"
        else:
            latest_landmarks = []
            predicted_label = "No Hand"

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)