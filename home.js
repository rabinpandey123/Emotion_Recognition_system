import React from 'react';
import '../styles/home.css';

const Home = () => {
  return (
    <div className="home-container">
      <div className="hero-section">
        <h1>Emotion Detection System</h1>
        <p>Advanced facial emotion recognition using DeepFace and FER models</p>
      </div>
      
      <div className="features-section">
        <h2>Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Real-time Analysis</h3>
            <p>Analyze emotions in real-time using your webcam</p>
          </div>
          <div className="feature-card">
            <h3>Dual Model Support</h3>
            <p>Utilizes both DeepFace and FER models for improved accuracy</p>
          </div>
          <div className="feature-card">
            <h3>Age & Gender Detection</h3>
            <p>Get additional insights with age and gender estimation</p>
          </div>
          <div className="feature-card">
            <h3>Camera Diagnostics</h3>
            <p>Built-in tools to troubleshoot camera issues</p>
          </div>
        </div>
      </div>
      
      <div className="how-to-section">
        <h2>How to Use</h2>
        <ol>
          <li>Navigate to the Emotion Detector tab</li>
          <li>Allow camera access when prompted</li>
          <li>If camera doesn't appear, use the camera diagnostics tool</li>
          <li>Click "Start Analysis" to begin emotion detection</li>
          <li>View results from both DeepFace and FER models</li>
        </ol>
      </div>
    </div>
  );
};

export default Home;