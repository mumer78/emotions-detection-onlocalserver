import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load your trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "emotion_model_save.keras")
model = load_model(MODEL_PATH)

# Map numbers to emotion labels
emotion_map = {0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise"}

# Load OpenCV's face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]

        # Preprocess face
        face_region = cv2.resize(face_region, (48, 48))
        face_region = face_region.astype("float32") / 255.0
        face_region = np.expand_dims(face_region, axis=0)    # batch dimension
        face_region = np.expand_dims(face_region, axis=-1)   # channel dimension

        # Predict emotion
        pred = model.predict(face_region, verbose=0)[0]
        max_idx = np.argmax(pred)
        emotion = emotion_map[max_idx]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display predicted emotion
        cv2.putText(frame, f"{emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show all probabilities
        for i, (emo, score) in enumerate(zip(emotion_map.values(), pred)):
            cv2.putText(frame, f"{emo}: {score:.2f}", (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()


