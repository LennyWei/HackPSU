import cv2
import pyautogui
import pygame
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
import threading
import time
from sklearn.model_selection import train_test_split
import json
import os
# testing blablbahl
BINDINGS_FILE = "gesture_bindings.json"
GESTURE_FOLDER = 'gestures'
MODEL_FOLDER = 'models'
os.makedirs(GESTURE_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


def normalize_landmarks(landmarks):
    base_x, base_y, _ = landmarks[0]  # Ignore z axis
    normalized_landmarks = [(x - base_x, y - base_y) for x, y, _ in landmarks]
    max_value = max([max(abs(x), abs(y)) for x, y in normalized_landmarks])
    if max_value > 0:
        normalized_landmarks = [(x / max_value, y / max_value) for x, y in normalized_landmarks]
    return normalized_landmarks

def load_bindings():
    if os.path.exists(BINDINGS_FILE):
        with open(BINDINGS_FILE, "r") as file:
            return json.load(file)
    return {}

latest_predictions = []  # Store predictions
latest_boxes = []  # Store boxes for each hand
latest_frames = None
last_prediction_time = 0
prediction_interval = 0.1  # Minimum interval between predictions (in seconds)
prediction_ready = threading.Event()  # Event to trigger predictions


prediction_lock = threading.Lock()

current_gesture = None
current_gesture_count = 0
gesture_accept_count = 5
bindings = load_bindings()

def predict_gesture(landmarks, model, label_map):
    normalized = normalize_landmarks(landmarks)
    prediction = model.predict(np.array([np.array(normalized).flatten()]))
    idx = np.argmax(prediction)
    label = list(label_map.keys())[list(label_map.values()).index(idx)]
    return label


def prediction_thread_function(model, label_map):
    global latest_frames, latest_predictions, latest_boxes, last_prediction_time
    global current_gesture, current_gesture_count
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        prediction_ready.wait()

        if latest_frames is None:
            prediction_ready.clear()
            continue

        frame_rgb = cv2.cvtColor(latest_frames, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()
        make_prediction = current_time - last_prediction_time >= prediction_interval

        if results.multi_hand_landmarks:
            current_predictions = []
            current_boxes = []

            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                x_vals = [lm.x for lm in hand_landmarks.landmark]
                y_vals = [lm.y for lm in hand_landmarks.landmark]

                # Only predict if enough time has passed
                if make_prediction:
                    label = predict_gesture(landmarks, model, label_map)
                else:
                    # Maintain previous label if we have one for this hand position
                    # This is simplified - in real code you'd want to track hands between frames
                    label = ""
                    if latest_predictions:
                        for prev_pred, prev_box in zip(latest_predictions, latest_boxes):
                            prev_x, prev_y = prev_box
                            # Check if this hand is close to a previous hand
                            if (abs(np.mean(x_vals) - np.mean(prev_x)) < 0.1 and 
                                    abs(np.mean(y_vals) - np.mean(prev_y)) < 0.1):
                                label = prev_pred['label']
                                break

                current_predictions.append({
                    'label': label,
                    'skeleton': hand_landmarks
                })
                current_boxes.append((x_vals, y_vals))

            # Update the last prediction time only when we actually make predictions
            # Input the gesture and connect it to the keybind
            if make_prediction:
                last_prediction_time = current_time 
                label = current_predictions[0]['label']

                # If gesture is same as before, increment counter
                if label == current_gesture:
                    current_gesture_count += 1
                else:
                    current_gesture = label
                    current_gesture_count = 1

                # If count threshold reached and gesture is bound to a key
                bindings = load_bindings()
                if (current_gesture_count == gesture_accept_count or current_gesture_count >= 20) and current_gesture in bindings:
                    print(f"Executing keybind for gesture: {current_gesture} -> {bindings[current_gesture]}")
                    pyautogui.press(bindings[current_gesture])


            with prediction_lock:
                latest_predictions = current_predictions
                latest_boxes = current_boxes





            # If no hands are detected, we should clear predictions
        elif make_prediction:
            with prediction_lock:
                latest_predictions = []
                latest_boxes = []
                last_prediction_time = current_time

        prediction_ready.clear()



def prediction_mode():
    global latest_predictions, latest_boxes, latest_frames

    print("Prediction Mode")
    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'gesture_model.h5'))
    with open(os.path.join(MODEL_FOLDER, 'label_map.json'), 'r') as f:
        label_map = json.load(f)

    cap = cv2.VideoCapture(0)

    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Prediction Mode")
    clock = pygame.time.Clock()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize global variables
    latest_predictions = []
    latest_boxes = []

    prediction_thread = threading.Thread(target=prediction_thread_function, args=(model, label_map), daemon=True)
    prediction_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        latest_frames = frame.copy()
        prediction_ready.set()

        frame_output = frame.copy()

        with prediction_lock:
            # Draw boxes and labels for all detected hands
            for i, (pred, (x_vals, y_vals)) in enumerate(zip(latest_predictions, latest_boxes)):
                label = pred['label']
                skeleton = pred['skeleton']
                h, w, _ = frame.shape
                x_min = int(min(x_vals) * w)
                x_max = int(max(x_vals) * w)
                y_min = int(min(y_vals) * h)
                y_max = int(max(y_vals) * h)

                # Always draw the bounding box, regardless of whether we have a label
                box_color = (0, 255, 0) if label else (255, 0, 0)  # Green if labeled, red if not
                cv2.rectangle(frame_output, (x_min, y_min), (x_max, y_max), box_color, 2)

                if label:  # Only draw the label text if available
                    cv2.putText(frame_output, label, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

                # Draw hand skeleton
                mp_drawing.draw_landmarks(frame_output, skeleton, mp_hands.HAND_CONNECTIONS)

        # Show frame counter and number of detected hands
        # cv2.putText(frame_output, f"Hands: {len(latest_predictions)}", (10, 30), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
        frame_output = np.fliplr(frame_output)
        frame_output = np.rot90(frame_output)
        frame_surface = pygame.surfarray.make_surface(frame_output)
        screen.blit(frame_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                cap.release()
                pygame.quit()
                return




def training_mode():
    print("Training Mode")
    X, y, label_map = [], [], {}
    for idx, gesture in enumerate(os.listdir(GESTURE_FOLDER)):
        label_map[gesture] = idx
        folder_path = os.path.join(GESTURE_FOLDER, gesture)
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                with open(os.path.join(folder_path, file), 'r') as f:
                    data = json.load(f)
                    X.append(np.array(data).flatten())
                    y.append(idx)

    if len(set(y)) < 2:
        print("Need at least two gestures to train.")
        return

    X, y = np.array(X), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    model.save(os.path.join(MODEL_FOLDER, 'gesture_model.h5'))
    with open(os.path.join(MODEL_FOLDER, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    print("Model saved to models/gesture_model.h5")



def data_collection_mode():
    print("Data Collection Mode")
    gesture_name = input("Enter gesture name or select an existing one (leave blank to choose from list): ").strip()

    if not gesture_name:
        gestures = os.listdir(GESTURE_FOLDER)
        if len(gestures) == 0:
            print("No gestures found. Please create a new one.")
            return

        print("Available gestures:")
        for i, gesture in enumerate(gestures, 1):
            print(f"{i}. {gesture}")
        try:
            choice = int(input("Select a gesture by number: "))
            gesture_name = gestures[choice - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Returning to main menu.")
            return

    gesture_path = os.path.join(GESTURE_FOLDER, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Data Collection Mode - {gesture_name}")
    clock = pygame.time.Clock()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_output = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized_landmarks = normalize_landmarks(landmarks)

                # Draw landmarks on frame using MediaPipe
                mp_drawing.draw_landmarks(frame_output, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:  # Save landmarks when Spacebar is pressed
                    file_path = os.path.join(gesture_path, f"landmarks_{len(os.listdir(gesture_path))}.json")
                    with open(file_path, 'w') as f:
                        json.dump(normalized_landmarks, f)
                    print(f"Landmarks saved to {file_path}")

        # Convert and display frame in Pygame
        frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
        frame_output = np.fliplr(frame_output)  # Mirror effect
        frame_output = np.rot90(frame_output)
        frame_surface = pygame.surfarray.make_surface(frame_output)
        screen.blit(frame_surface, (0, 0))

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                cap.release()
                hands.close()
                pygame.quit()
                return

def cli_menu():
    while True:
        print("\nGesture Recognition CLI")
        print("1. Data Collection Mode")
        print("2. Training Mode")
        print("3. Prediction Mode")
        print("4. Enter Binding Mode")
        print("5. Exit")
        choice = input("Select an option (1-4): ")

        if choice == '1':
            data_collection_mode()
        elif choice == '2':
            training_mode()
        elif choice == '3':
            prediction_mode()
        elif choice == '4':
            gesture_binding_cli()
        elif choice == '5':    
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

def save_bindings(bindings):
    with open(BINDINGS_FILE, "w") as file:
        json.dump(bindings, file, indent=4)

def gesture_binding_cli():
    bindings = load_bindings()

    while True:
        print("\n--- Gesture to Keybind CLI ---")
        print("1. View current bindings (table view)")
        print("2. Add/Edit a binding")
        print("3. Delete a binding")
        print("4. Save and exit")
        print("5. Exit without saving")
        choice = input("Choose an option: ")        
        if choice == "1":
            if not bindings:
                print("No bindings found.")
            else:
                for gesture, key in bindings.items():
                    print()
                    print(f"{gesture} -> {key}")
        elif choice == "2":
            gestures = os.listdir(GESTURE_FOLDER)
            if not gestures:
                print("No gestures found. Please collect gesture data first.")
                continue

            print("Available gestures:")
            for i, g in enumerate(gestures, 1):
                print(f"{i}. {g}")

            gesture = input("Enter gesture name exactly as shown: ").strip()
            if gesture not in gestures:
                print("Invalid gesture name.")
                continue
            key = input("Enter key to bind (e.g., left, right, enter): ")
            bindings[gesture] = key
            print(f"Bound '{gesture}' to '{key}'")
        elif choice == "3":
            if not bindings:
                print("No bindings to delete.")
                continue

            print("\nBindings:")
            for i, gesture in enumerate(bindings.keys(), 1):
                print(f"{i}. {gesture} -> {bindings[gesture]}")

            try:
                index = int(input("Select a binding to delete by number: ").strip())
                gesture = list(bindings.keys())[index - 1]
                del bindings[gesture]
                print(f"Deleted binding for '{gesture}'.")
            except (ValueError, IndexError):
                print("Invalid selection.")

        elif choice == "4":
            save_bindings(bindings)
            print("Bindings saved. Exiting.")
            break

        elif choice == "5":
            print("Exiting without saving.")
            break

        else:
            print("Invalid option. Try again.")



if __name__ == "__main__":
    cli_menu()


