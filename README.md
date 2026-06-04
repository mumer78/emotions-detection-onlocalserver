# 🧠 Live Emotion Detection — Local Server

> Real-time facial emotion detection using a custom-trained CNN model, Flask backend, and React frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat&logo=flask)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat&logo=opencv)

---

## ✨ Core Features

- 🎥 **Live webcam feed** — detects faces and classifies emotions in real time
- 🤖 **Custom CNN model** — trained from scratch on a facial expression dataset
- 😀 **5 Emotions detected** — Angry, Happy, Neutral, Sad, Surprise
- 📊 **Probability bars** — shows confidence percentage for each emotion live
- ⚡ **Instant response** — runs locally so zero network latency
- 🧹 **Memory optimized** — garbage collection after each frame prediction

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | CNN (TensorFlow / Keras) |
| Face Detection | OpenCV Haar Cascade |
| Backend | Flask + Flask-CORS |
| Frontend | React + Vite + Tailwind CSS |
| Image Processing | NumPy, OpenCV, Base64 |

---

## 📁 Project Structure

```
Emotion-detection-localserver/
├── emotion_model_save.keras      # Trained CNN model
├── code/
│   ├── app.py                    # Flask backend
│   └── haarcascade_frontalface_default.xml
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── ...
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Getting Started

> ⚠️ You need **two terminals** running at the same time — one for the backend, one for the frontend.

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### 1️⃣ Clone the repo

```bash
git clone https://github.com/mumer78/emotions-detection-onlocalserver.git
cd emotions-detection-onlocalserver
```

### 2️⃣ Terminal 1 — Start Flask Backend

```bash
cd code
pip install flask flask-cors tensorflow opencv-python numpy
python app.py
```

You should see:
```
SUCCESS: Loaded cascade classifier
SUCCESS: Model warmed up
* Running on http://127.0.0.1:5000
```

### 3️⃣ Terminal 2 — Start React Frontend

Open a **new terminal** (keep Terminal 1 running):

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
VITE v6.x  ready in 300ms
➜  Local:   http://localhost:5173/
```

### 4️⃣ Open in browser

```
http://localhost:5173
```

Allow camera access when prompted — emotion detection starts instantly! 🎉

---

## ⚙️ How It Works

```
Webcam (Browser)
      ↓  captures frame every 100ms
React Frontend (port 5173)
      ↓  sends base64 image via POST /api/predict
Flask Backend (port 5000)
      ↓  detects face with Haar Cascade
      ↓  resizes to 48x48 grayscale
      ↓  runs CNN model inference
      ↓  returns emotion + probabilities
React Frontend
      ↓  draws bounding box + emotion label
You see the result live ✅
```

---

## 🧠 Model Details

- Architecture: **Custom CNN** built with TensorFlow/Keras
- Input: **48x48 grayscale** facial image
- Output: **5 emotion classes** — Angry, Happy, Neutral, Sad, Surprise
- Training: Data augmentation + Dropout regularization for generalization
- Format: `.keras`

---

## ❓ Troubleshooting

**`ECONNREFUSED 127.0.0.1:5000`**
> Flask backend is not running. Start Terminal 1 first.

**`Failed to load cascade classifier`**
> Make sure `haarcascade_frontalface_default.xml` is inside the `code/` folder.

**`Model not found`**
> Make sure `emotion_model_save.keras` is in the root of the project (not inside `code/`).

**Slow detection**
> Make sure you are running locally and NOT on a cloud server. This repo is optimized for local use only.

---

## 🌐 Cloud Version (Slower)

If you want to try the deployed version (Render free tier — expect delays due to CPU limits):

- 🔗 Live Demo: [face-emotion-detectionl.vercel.app](https://face-emotion-detectionl.vercel.app/)
- 📁 Cloud Code: [github.com/mumer78/Emotion-detection](https://github.com/mumer78/Emotion-detection)

---

## 👨‍💻 Author

**Muhammad Umer**
- 🌐 Portfolio: [portfolio-muhammad-umer.vercel.app](https://portfolio-muhammad-umer.vercel.app/)
- 💼 LinkedIn: [linkedin.com/in/muhammad-umer-247970318](https://linkedin.com/in/muhammad-umer-247970318)
- 🐙 GitHub: [github.com/mumer78](https://github.com/mumer78)

---

> 🎓 Built as part of my **4th Semester AI Course** at Forman Christian College, Lahore.
