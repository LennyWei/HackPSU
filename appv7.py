from flask import Flask, render_template, request, jsonify, Response
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import threading
import cv2
import mediapipe as mp
import time
import pyautogui
import shutil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_FOLDER = os.path.join(BASE_DIR, 'HackPSU', 'gestures')
MODEL_PATH = 'models/gesture_model.h5'
LABEL_MAP_PATH = 'models/label_map.json'
BINDINGS_FILE = 'gesture_bindings.json'

model = None
label_map = {}
model_lock = threading.Lock()
latest_prediction = "Waiting..."
latest_landmarks = []
current_gesture = None
current_gesture_count = 0
GESTURE_HOLD_COUNT = 3

# Load model and label map
if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

desired_width = 1920 
desired_height = 1080 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": latest_prediction})

@app.route('/get_landmarks')
def get_landmarks():
    if latest_landmarks:
        return jsonify({"landmarks": latest_landmarks})
    return jsonify({"error": "No landmarks detected"})

@app.route('/data-collection')
def data_collection():
    return render_template('data-collection.html')

@app.route('/contact')
def contact():
    return render_template('page.html', title='Contact')

@app.route('/collect_batch', methods=['POST'])
def collect_batch():
    data = request.json
    gesture = data.get('gesture')
    delay = data.get('delay', 0.05)  # seconds
    count = data.get('count', 100)

    if not gesture:
        return jsonify({"message": "Gesture name required"}), 400

    gesture_path = os.path.join(GESTURE_FOLDER, gesture)
    os.makedirs(gesture_path, exist_ok=True)

    collected = 0
    while collected < count:
        if latest_landmarks:
            file_path = os.path.join(gesture_path, f"landmarks_{len(os.listdir(gesture_path))}.json")
            with open(file_path, 'w') as f:
                json.dump(latest_landmarks, f)
            collected += 1
            print(f"[INFO] Saved frame {collected}/{count}")
        time.sleep(delay)

    return jsonify({"message": f"{collected} frames saved for gesture '{gesture}'"})
     
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

@app.route('/test')
def test():
    return render_template('video_feed.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/gestures')
def gestures():
    gestures_list = sorted(os.listdir(GESTURE_FOLDER)) if os.path.exists(GESTURE_FOLDER) else []
    bindings = load_bindings()
    return render_template('gestures.html', gestures_list=gestures_list, bindings=bindings)

@app.route('/get_bindings')
def get_bindings():
    return jsonify(load_bindings())

@app.route('/bind_gesture', methods=['POST'])
def bind_gesture():
    gesture = request.form.get('gesture')
    key = request.form.get('key')
    if not gesture or not key:
        return jsonify({'error': 'Missing gesture or key'}), 400

    bindings = load_bindings()
    bindings[gesture] = key
    save_bindings(bindings)

    gestures_list = sorted(os.listdir(GESTURE_FOLDER)) if os.path.exists(GESTURE_FOLDER) else []
    return render_template('gestures.html', gestures_list=gestures_list, bindings=bindings)  


@app.route('/delete_binding', methods=['POST'])
def delete_binding():
    gesture = request.form.get('gesture')
    bindings = load_bindings()
    if gesture in bindings:
        del bindings[gesture]
        save_bindings(bindings)

    gesture_path = os.path.join(GESTURE_FOLDER, gesture)
    if os.path.exists(gesture_path) and os.path.isdir(gesture_path):
        shutil.rmtree(gesture_path)

    gestures_list = sorted(os.listdir(GESTURE_FOLDER)) if os.path.exists(GESTURE_FOLDER) else []
    return render_template('gestures.html', gestures_list=gestures_list, bindings=bindings)


def load_bindings():
    if os.path.exists(BINDINGS_FILE):
        with open(BINDINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_bindings(bindings):
    with open(BINDINGS_FILE, 'w') as f:
        json.dump(bindings, f, indent=4)

def generate_video():
    global latest_prediction, latest_landmarks, current_gesture, current_gesture_count
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
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            base_x, base_y, _ = landmarks[0]
            norm = [(x - base_x, y - base_y) for x, y, _ in landmarks]
            max_val = max([max(abs(x), abs(y)) for x, y in norm])
            if max_val > 0:
                norm = [(x / max_val, y / max_val) for x, y in norm]
            latest_landmarks = norm

            if model is not None and label_map:
                input_data = np.array([np.array(norm).flatten()])
                with model_lock:
                    prediction = model.predict(input_data, verbose=0)
                    predicted_index = np.argmax(prediction)
                    predicted_label = list(label_map.keys())[list(label_map.values()).index(predicted_index)]

                # Gesture consistency
                if predicted_label == current_gesture:
                    current_gesture_count += 1
                else:
                    current_gesture = predicted_label
                    current_gesture_count = 1

                bindings = load_bindings()
                if (current_gesture_count == GESTURE_HOLD_COUNT  or current_gesture_count >= 15) and predicted_label in bindings:
                    pyautogui.press(bindings[predicted_label])
                    current_gesture_count = 0  # reset after triggering

                latest_prediction = predicted_label
            else:
                latest_prediction = "Hand Detected"
        else:
            latest_landmarks = []
            latest_prediction = "No Hand"
        frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
    
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

def train_and_reload(X_train, y_train, X_val, y_val, label_map):
    global model
    with model_lock:
        local_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(label_map), activation='softmax')
        ])
        local_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        local_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

        local_model.save(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'w') as f:
            json.dump(label_map, f)

        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map.update(json.load(f))

@app.route('/train_model', methods=['POST'])
def train_model():
    global model, label_map

    X, y = [], []
    label_map = {}

    for idx, gesture in enumerate(os.listdir(GESTURE_FOLDER)):
        gesture_path = os.path.join(GESTURE_FOLDER, gesture)
        label_map[gesture] = idx
        for file in os.listdir(gesture_path):
            if file.endswith('.json'):
                with open(os.path.join(gesture_path, file), 'r') as f:
                    data = json.load(f)
                    X.append(np.array(data).flatten())
                    y.append(idx)

    if len(set(y)) < 2:
        return jsonify({"message": "Need at least two gesture classes to train."}), 400

    X = np.array(X)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    thread = threading.Thread(target=train_and_reload, args=(X_train, y_train, X_val, y_val, label_map))
    thread.start()

    return jsonify({"message": "Model training started in background."})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
