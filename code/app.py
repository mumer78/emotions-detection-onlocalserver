from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import gc

# Limit TensorFlow memory before importing keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, max_age=86400)

# ------------------------------
# Load emotion detection model
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "emotion_model_save.keras")
model = load_model(MODEL_PATH)
emotion_map = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprise"}

# ------------------------------
# Load face detector
# ------------------------------
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
if face_detector.empty():
    print(f"ERROR: Failed to load cascade classifier from {CASCADE_PATH}")
else:
    print(f"SUCCESS: Loaded cascade classifier from {CASCADE_PATH}")

# ------------------------------
# Warm up the model
# ------------------------------
dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
_ = model(dummy, training=False)
del dummy
gc.collect()
print("SUCCESS: Model warmed up")


# ------------------------------
# Generate video frames (local webcam)
# ------------------------------
def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1).astype(np.float32)

            pred = model(face, training=False).numpy()[0]
            emotion = emotion_map[np.argmax(pred)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            for i, (emo, prob) in enumerate(zip(emotion_map.values(), pred)):
                cv2.putText(
                    frame,
                    f"{emo}: {prob*100:.2f}%",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ------------------------------
# Flask routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data["image"]
        if "," in image_data:
            header, image_data = image_data.split(",", 1)

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for x, y, w, h in faces:
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype(np.float32) / 255.0
            face = face.reshape(1, 48, 48, 1)

            pred = model(face, training=False).numpy()[0]
            max_idx = np.argmax(pred)
            emotion = emotion_map[max_idx]

            probabilities = {}
            for i, emo in emotion_map.items():
                probabilities[emo] = float(pred[i])

            results.append(
                {
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "probabilities": probabilities,
                }
            )

        # Free memory
        del frame, gray, np_array, image_bytes
        gc.collect()

        return jsonify({"faces": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=False, use_reloader=False)
