
import cv2
import pygame
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
from sklearn.model_selection import train_test_split

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




def prediction_mode():
    print("Prediction Mode")
    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'gesture_model.h5'))
    with open(os.path.join(MODEL_FOLDER, 'label_map.json'), 'r') as f:
        label_map = json.load(f)  # Load label map as-is, without converting keys to integers

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Prediction Mode")
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
                mp_drawing.draw_landmarks(frame_output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized = normalize_landmarks(landmarks)
                prediction = model.predict(np.array([np.array(normalized).flatten()]))
                idx = np.argmax(prediction)
                label = list(label_map.keys())[list(label_map.values()).index(idx)]

                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                cv2.rectangle(frame_output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_output, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
        frame_output = np.fliplr(frame_output)
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
        print("4. Exit")
        choice = input("Select an option (1-4): ")

        if choice == '1':
            data_collection_mode()
        elif choice == '2':
            training_mode()
        elif choice == '3':
            prediction_mode()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    cli_menu()


