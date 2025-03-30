from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import json
import numpy as np
import threading
import os
import pyautogui
import time


app = Flask(__name__)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

GESTURE_FOLDER = 'gestures'
MODEL_PATH = 'models/gesture_model.h5'
LABEL_MAP_PATH = 'models/label_map.json'
BINDINGS_FILE = 'gesture_bindings.json'

latest_frame = None
predicted_gesture = "Waiting..."
predicted_counter = 0
prediction_lock = threading.Lock()
predict_ready = threading.Event()

# Load label map and model if available
label_map = {}
model = None
try:
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
        model = load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
except Exception as e:
    print("Model not loaded:", e)

# Load gesture bindings
bindings = {}
if os.path.exists(BINDINGS_FILE):
    with open(BINDINGS_FILE, 'r') as f:
        bindings = json.load(f)

# Gesture debounce variables
current_gesture = None
current_gesture_count = 0
gesture_accept_count = 5
last_prediction_time = 0
prediction_interval = 0.1  # in seconds

def normalize_landmarks(landmarks):
    base_x, base_y = landmarks[0]
    normalized = [(x - base_x, y - base_y) for x, y in landmarks]
    max_val = max(max(abs(x), abs(y)) for x, y in normalized)
    if max_val > 0:
        normalized = [(x / max_val, y / max_val) for x, y in normalized]
    return normalized

def load_bindings():
    if os.path.exists(BINDINGS_FILE):
        with open(BINDINGS_FILE, "r") as file:
            return json.load(file)
    return {}

def prediction_worker():
    global predicted_gesture, current_gesture, current_gesture_count, predicted_counter, last_prediction_time, bindings
    while True:
        predict_ready.wait()
        frame = latest_frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()
        make_prediction = current_time - last_prediction_time >= prediction_interval

        if results.multi_hand_landmarks and model is not None and make_prediction:
            landmarks = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]
            normalized = normalize_landmarks(landmarks)
            prediction = model.predict(np.array([np.array(normalized).flatten()]), verbose=0)
            idx = np.argmax(prediction)
            label = list(label_map.keys())[list(label_map.values()).index(idx)]

            last_prediction_time = current_time

            # Gesture consistency check
            if label == current_gesture:
                current_gesture_count += 1
            else:
                current_gesture = label
                current_gesture_count = 1

            bindings = load_bindings()
            if (current_gesture_count == gesture_accept_count or current_gesture_count >= 20) and current_gesture in bindings:
                print(f"Executing keybind for gesture: {current_gesture} -> {bindings[current_gesture]}")
                pyautogui.press(bindings[current_gesture])

            with prediction_lock:
                predicted_gesture = label
                predicted_counter = current_gesture_count

        predict_ready.clear()

# Start background prediction thread
threading.Thread(target=prediction_worker, daemon=True).start()

def generate():
    global latest_frame
    while True:
        success, frame = cap.read()
        if not success:
            break

        latest_frame = frame.copy()
        predict_ready.set()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        with prediction_lock:
            label = predicted_gesture
            counter = predicted_counter

        cv2.putText(frame, f"Gesture: {label} ({counter})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with prediction_lock:
        return jsonify({"prediction": predicted_gesture, "count": predicted_counter})

@app.route("/train")
def train():
    return render_template("page.html", title="Train")

@app.route("/test")
def test():
    return render_template("page.html", title="Test")

@app.route("/data-collection")
def data_collection():
    return render_template("page.html", title="Data Collection")

@app.route("/contact")
def contact():
    return render_template("page.html", title="Contact")

@app.route("/gestures")
def gestures():
    return render_template("page.html", title="Gestures")

@app.route('/live')
def live():
    return render_template('video_feed.html')

if __name__ == '__main__':
    try:
        app.run(debug=True, threaded=True, use_reloader=False)
    finally:
        cap.release()
        cv2.destroyAllWindows()
