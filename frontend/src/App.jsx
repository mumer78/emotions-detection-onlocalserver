import React, { useState, useEffect, useRef } from 'react';
import { Camera, Sparkles, Smile, X, Activity, Video, AlertCircle, RefreshCw } from 'lucide-react';

function App() {
  const [isActive, setIsActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [faces, setFaces] = useState([]);
  const [stats, setStats] = useState({
    fps: 0,
    facesCount: 0,
    confidence: 0,
    primaryEmotion: 'None',
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const lastFrameTimeRef = useRef(Date.now());
  const isProcessingRef = useRef(false);

  // Map emotions to colors and emojis
  const emotionConfig = {
    Happy: { emoji: '😊', color: 'var(--color-happy)', bg: 'rgba(234, 179, 8, 0.15)' },
    Sad: { emoji: '😢', color: 'var(--color-sad)', bg: 'rgba(59, 130, 246, 0.15)' },
    Angry: { emoji: '😠', color: 'var(--color-angry)', bg: 'rgba(239, 68, 68, 0.15)' },
    Surprise: { emoji: '😮', color: 'var(--color-surprise)', bg: 'rgba(168, 85, 247, 0.15)' },
    Neutral: { emoji: '😐', color: 'var(--color-neutral)', bg: 'rgba(156, 163, 175, 0.15)' },
  };

  // Start webcam stream
  const startCamera = async () => {
    setLoading(true);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      setIsActive(true);
      setLoading(false);
    } catch (err) {
      console.error("Camera access error:", err);
      setError("Unable to access camera. Please check your permissions.");
      setLoading(false);
    }
  };

  // Stop webcam stream
  const stopCamera = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsActive(false);
    setFaces([]);
    setStats({
      fps: 0,
      facesCount: 0,
      confidence: 0,
      primaryEmotion: 'None',
    });
  };

  // Attach stream to video element once it is rendered in the DOM
  useEffect(() => {
    if (isActive && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play().catch((err) => console.error("Video play failed:", err));
        startDetectionLoop();
      };
    }
  }, [isActive]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Frame processing and prediction loop
  const startDetectionLoop = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    // Create a hidden canvas for capturing frames at 320x240 (4x fewer pixels, much faster detection & upload)
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 320;
    captureCanvas.height = 240;
    const captureContext = captureCanvas.getContext('2d');

    intervalRef.current = setInterval(async () => {
      if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;
      if (isProcessingRef.current) return;

      isProcessingRef.current = true;

      // Draw current video frame to hidden canvas at 320x240
      captureContext.drawImage(videoRef.current, 0, 0, 320, 240);
      const base64Image = captureCanvas.toDataURL('image/jpeg', 0.5);

      try {
        const apiUrl = import.meta.env.VITE_API_URL 
          ? `${import.meta.env.VITE_API_URL}/api/predict` 
          : '/api/predict';

        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) throw new Error("API response error");

        const data = await response.json();
        
        // Calculate FPS
        const now = Date.now();
        const fps = Math.round(1000 / (now - lastFrameTimeRef.current));
        lastFrameTimeRef.current = now;

        if (data.faces && data.faces.length > 0) {
          setFaces(data.faces);
          
          // Compute summary stats
          const mainFace = data.faces[0];
          const emotion = mainFace.emotion;
          const confidence = Math.round(mainFace.probabilities[emotion] * 100);

          setStats({
            fps,
            facesCount: data.faces.length,
            confidence,
            primaryEmotion: emotion,
          });
        } else {
          setFaces([]);
          setStats((prev) => ({
            ...prev,
            fps,
            facesCount: 0,
            confidence: 0,
            primaryEmotion: 'None',
          }));
        }
      } catch (err) {
        console.error("Prediction loop error:", err);
      } finally {
        isProcessingRef.current = false;
      }
    }, 200); // 5 times per second (fast, smooth, and lightweight)
  };

  // Draw bounding boxes on the overlay canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isActive) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      const [x, y, w, h] = face.box;
      
      // Scale bounding box coordinates from 320x240 to canvas dimensions (usually 640x480)
      const scaleX = canvas.width / 320;
      const scaleY = canvas.height / 240;
      
      const xScaled = x * scaleX;
      const yScaled = y * scaleY;
      const wScaled = w * scaleX;
      const hScaled = h * scaleY;

      // Calculate mirrored x coordinate because video is scaleX(-1) but canvas is not
      const xMirrored = canvas.width - xScaled - wScaled;
      
      const emotion = face.emotion;
      const config = emotionConfig[emotion] || emotionConfig.Neutral;

      // Draw bounding box
      ctx.strokeStyle = config.color;
      ctx.lineWidth = 4;
      ctx.lineJoin = 'round';
      ctx.strokeRect(xMirrored, yScaled, wScaled, hScaled);

      // Draw shadow background for label
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(xMirrored, yScaled - 35, wScaled, 35);

      // Draw text label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 16px Poppins, sans-serif';
      const labelText = `${config.emoji} ${emotion}`;
      ctx.fillText(labelText, xMirrored + 10, yScaled - 11);
    });
  }, [faces, isActive]);

  // Main active face details
  const activeFace = faces[0] || null;

  return (
    <div className="app-container">
      {/* Background gradients */}
      <div className="bg-glow bg-glow-1"></div>
      <div className="bg-glow bg-glow-2"></div>

      {/* Header */}
      <header className="app-header">
        <div className="logo-section">
          <Smile className="logo-icon" />
          <span>Emotion<span className="accent">AI</span></span>
        </div>
        <div className="status-badge">
          <div className={`status-dot ${isActive ? 'active' : ''}`}></div>
          <span>{isActive ? 'System Live' : 'System Idle'}</span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="main-content">
        {!isActive ? (
          /* Landing Screen */
          <div className="landing-card animate-fade-in">
            <div className="icon-wrapper">
              <Sparkles className="sparkle-icon" />
            </div>
            <h1>Reveal Your Emotions In Real-Time</h1>
            <p>
              Experience state-of-the-art Convolutional Neural Network (CNN) emotion recognition. 
              Our AI instantly analyzes your facial micro-expressions directly in the browser.
            </p>

            {error && (
              <div className="error-message">
                <AlertCircle className="error-icon" />
                <span>{error}</span>
              </div>
            )}

            <button 
              className={`btn-start ${loading ? 'loading' : ''}`}
              onClick={startCamera}
              disabled={loading}
              id="start-detection-btn"
            >
              {loading ? (
                <>
                  <RefreshCw className="spinner" />
                  Starting Camera...
                </>
              ) : (
                <>
                  <Camera className="btn-icon" />
                  Start Live Detection
                </>
              )}
            </button>
          </div>
        ) : (
          /* Live Detection Dashboard */
          <div className="dashboard-grid animate-scale-up">
            {/* Left: Video Feed Card */}
            <div className="panel video-panel">
              <div className="panel-header">
                <div className="panel-title">
                  <Video className="icon-green" />
                  <h2>Live Webcam Stream</h2>
                </div>
                <button className="btn-close" onClick={stopCamera}>
                  <X size={20} />
                </button>
              </div>

              <div className="video-container">
                <video
                  ref={videoRef}
                  width="640"
                  height="480"
                  autoplay
                  playsinline
                  muted
                ></video>
                <canvas
                  ref={canvasRef}
                  width="640"
                  height="480"
                  className="overlay-canvas"
                ></canvas>
              </div>

              <button className="btn-stop" onClick={stopCamera}>
                <X size={18} /> Stop Detecting
              </button>
            </div>

            {/* Right: Analytics Panel */}
            <div className="panel stats-panel">
              <div className="panel-header">
                <div className="panel-title">
                  <Activity className="icon-green" />
                  <h2>Real-Time Analytics</h2>
                </div>
              </div>

              {/* Stats Counters */}
              <div className="stats-row">
                <div className="stat-card">
                  <span className="stat-label">Faces</span>
                  <span className="stat-value">{stats.facesCount}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">FPS</span>
                  <span className="stat-value">{stats.fps}</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Confidence</span>
                  <span className="stat-value">{stats.confidence}%</span>
                </div>
              </div>

              {/* Primary Emotion Highlight */}
              <div 
                className="primary-emotion-card" 
                style={{ 
                  backgroundColor: stats.primaryEmotion !== 'None' 
                    ? emotionConfig[stats.primaryEmotion]?.bg 
                    : 'rgba(255, 255, 255, 0.03)' 
                }}
              >
                <span className="label">Detected Emotion</span>
                <div className="value-group">
                  <span className="emoji">
                    {stats.primaryEmotion !== 'None' ? emotionConfig[stats.primaryEmotion]?.emoji : '🔍'}
                  </span>
                  <span 
                    className="text" 
                    style={{ color: emotionConfig[stats.primaryEmotion]?.color || '#fff' }}
                  >
                    {stats.primaryEmotion}
                  </span>
                </div>
              </div>

              {/* Probability Distribution */}
              <div className="distribution-section">
                <h3>Emotion Probabilities</h3>
                <div className="progress-bars-container">
                  {Object.entries(emotionConfig).map(([emotion, config]) => {
                    const prob = activeFace ? activeFace.probabilities[emotion] || 0 : 0;
                    const pct = Math.round(prob * 100);

                    return (
                      <div className="progress-item" key={emotion}>
                        <div className="progress-labels">
                          <span>{config.emoji} {emotion}</span>
                          <span>{pct}%</span>
                        </div>
                        <div className="progress-track">
                          <div 
                            className="progress-fill" 
                            style={{ 
                              width: `${pct}%`, 
                              backgroundColor: config.color 
                            }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        Powered by OpenCV, TensorFlow & Flask • Built for Web Browsers
      </footer>
    </div>
  );
}

export default App;
