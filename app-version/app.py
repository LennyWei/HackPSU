from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

app = Flask(__name__)

# Load model and label map
model = tf.keras.models.load_model("models/gesture_model.h5")
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    base_x, base_y, _ = landmarks[0]
    normalized = [(x - base_x, y - base_y) for x, y, _ in landmarks]
    max_value = max([max(abs(x), abs(y)) for x, y in normalized])
    return [(x / max_value, y / max_value) for x, y in normalized] if max_value > 0 else normalized

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized = normalize_landmarks(landmarks)
                prediction = model.predict(np.array([np.array(normalized).flatten()]))
                idx = np.argmax(prediction)
                label = list(label_map.keys())[list(label_map.values()).index(idx)]

                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/livestream")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
