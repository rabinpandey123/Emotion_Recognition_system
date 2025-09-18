import React, { useState, useRef, useEffect } from 'react';
import '../styles/EmotionDetector.css';

const EmotionDetector = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [results, setResults] = useState(null);
  const [videoMode, setVideoMode] = useState(false);
  const [cameraDiagnostics, setCameraDiagnostics] = useState(false);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [ferDiagnostics, setFerDiagnostics] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const analysisIntervalRef = useRef(null);

  // Get available camera devices
  useEffect(() => {
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setCameraDevices(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (error) {
        console.error('Error getting camera devices:', error);
      }
    };

    getCameras();
  }, []);

  // Initialize camera
  const initCamera = async (deviceId = null) => {
    try {
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints = deviceId 
        ? { video: { deviceId: { exact: deviceId } } }
        : { video: true };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      setCameraError(null);
      return true;
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError(`Camera error: ${error.message}`);
      return false;
    }
  };

  // Capture frame from video
  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    return canvas.toDataURL('image/jpeg');
  };

  // Analyze image
  const analyzeImage = async () => {
    const imageData = captureFrame();
    if (!imageData) return;

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      if (data.success) {
        setResults(data);
      } else {
        console.error('Analysis error:', data.error);
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
    }
  };

  // Start video analysis
  const startVideoAnalysis = async () => {
    const success = await initCamera(selectedCamera || null);
    if (!success) return;

    setIsAnalyzing(true);
    setVideoMode(true);

    // Start periodic analysis
    analysisIntervalRef.current = setInterval(() => {
      analyzeImage();
    }, 1000); // Analyze every second
  };

  // Stop analysis
  const stopAnalysis = () => {
    setIsAnalyzing(false);
    setVideoMode(false);
    
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
    }
    
    // Stop the camera stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  // Run camera diagnostics
  const runCameraDiagnostics = async () => {
    setCameraDiagnostics(true);
    
    // Try to initialize camera with each available device
    for (const device of cameraDevices) {
      const success = await initCamera(device.deviceId);
      if (success) {
        // Wait a moment for the camera to initialize
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Try to capture a frame
        const frame = captureFrame();
        if (frame) {
          console.log(`Camera ${device.label} works correctly`);
        }
      }
    }
    
    // Reinitialize with the selected camera
    if (selectedCamera) {
      await initCamera(selectedCamera);
    }
  };

  // Run FER model diagnostics
  const runFerDiagnostics = async () => {
    const imageData = captureFrame();
    if (!imageData) return;

    try {
      const response = await fetch('http://localhost:5000/api/diagnose_fer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      const data = await response.json();
      if (data.success) {
        setFerDiagnostics(data);
      } else {
        console.error('FER diagnostics error:', data.error);
      }
    } catch (error) {
      console.error('Error running FER diagnostics:', error);
    }
  };

  // Draw bounding boxes on canvas
  useEffect(() => {
    if (!results || !canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;
    
    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw video frame
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes and labels for DeepFace results
    if (results.deepface && results.deepface.length > 0) {
      results.deepface.forEach((result, index) => {
        const [x, y, w, h] = result.bbox;
        
        // Draw bounding box
        context.strokeStyle = '#00ff00';
        context.lineWidth = 2;
        context.strokeRect(x, y, w, h);
        
        // Draw label background
        context.fillStyle = 'rgba(0, 0, 0, 0.7)';
        context.fillRect(x, y - 20, 150, 20);
        
        // Draw label text
        context.fillStyle = '#ffffff';
        context.font = '14px Arial';
        context.fillText(
          `DF: ${result.emotion} (${result.gender}, ${result.age})`, 
          x + 5, 
          y - 5
        );
      });
    }
    
    // Draw bounding boxes and labels for FER results
    if (results.fer && results.fer.length > 0) {
      results.fer.forEach((result, index) => {
        const [x, y, w, h] = result.bbox;
        
        // Draw bounding box
        context.strokeStyle = '#ff9900';
        context.lineWidth = 2;
        context.strokeRect(x, y, w, h);
        
        // Draw label background
        context.fillStyle = 'rgba(0, 0, 0, 0.7)';
        context.fillRect(x, y + h, 150, 20);
        
        // Draw label text
        context.fillStyle = '#ffffff';
        context.font = '14px Arial';
        context.fillText(
          `FER: ${result.emotion} (${result.confidence}%)`, 
          x + 5, 
          y + h + 15
        );
      });
    }
  }, [results]);

  return (
    <div className="emotion-detector-container">
      <h1>Emotion Detection</h1>
      
      <div className="camera-section">
        <div className="video-container">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className={cameraError ? 'camera-error' : ''}
          />
          <canvas ref={canvasRef} className="overlay-canvas" />
          {cameraError && (
            <div className="camera-error-message">
              <p>{cameraError}</p>
              <button onClick={() => initCamera(selectedCamera)}>Retry</button>
            </div>
          )}
        </div>
        
        <div className="camera-controls">
          <div className="camera-selection">
            <label htmlFor="camera-select">Select Camera:</label>
            <select 
              id="camera-select"
              value={selectedCamera}
              onChange={(e) => setSelectedCamera(e.target.value)}
              disabled={isAnalyzing}
            >
              {cameraDevices.map(device => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId}`}
                </option>
              ))}
            </select>
            <button 
              onClick={() => initCamera(selectedCamera)}
              disabled={isAnalyzing}
            >
              Switch Camera
            </button>
          </div>
          
          {!isAnalyzing ? (
            <button 
              className="start-button"
              onClick={startVideoAnalysis}
              disabled={cameraDevices.length === 0}
            >
              Start Analysis
            </button>
          ) : (
            <button 
              className="stop-button"
              onClick={stopAnalysis}
            >
              Stop Analysis
            </button>
          )}
        </div>
      </div>
      
      <div className="diagnostics-section">
        <h2>Diagnostics</h2>
        <div className="diagnostics-buttons">
          <button onClick={runCameraDiagnostics} disabled={isAnalyzing}>
            Camera Diagnostics
          </button>
          <button onClick={runFerDiagnostics} disabled={!isAnalyzing}>
            FER Model Diagnostics
          </button>
        </div>
        
        {ferDiagnostics && (
          <div className="fer-diagnostics">
            <h3>FER Model Detailed Predictions</h3>
            <div className="predictions-grid">
              {Object.entries(ferDiagnostics.predictions).map(([emotion, confidence]) => (
                <div key={emotion} className="prediction-item">
                  <span className="emotion-label">{emotion}:</span>
                  <div className="confidence-bar-container">
                    <div 
                      className="confidence-bar" 
                      style={{ width: `${confidence}%` }}
                    ></div>
                  </div>
                  <span className="confidence-value">{confidence.toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <p>
              Dominant Emotion: <strong>{ferDiagnostics.dominant_emotion}</strong> 
              (Confidence: {ferDiagnostics.confidence.toFixed(1)}%)
            </p>
          </div>
        )}
      </div>
      
      {results && (
        <div className="results-section">
          <h2>Analysis Results</h2>
          
          <div className="results-grid">
            <div className="result-card">
              <h3>DeepFace Results</h3>
              {results.deepface && results.deepface.length > 0 ? (
                results.deepface.map((result, index) => (
                  <div key={index} className="result-item">
                    <p><strong>Emotion:</strong> {result.emotion}</p>
                    <p><strong>Gender:</strong> {result.gender}</p>
                    <p><strong>Age:</strong> {result.age}</p>
                  </div>
                ))
              ) : (
                <p>No faces detected by DeepFace</p>
              )}
            </div>
            
            <div className="result-card">
              <h3>FER Results</h3>
              {results.fer && results.fer.length > 0 ? (
                results.fer.map((result, index) => (
                  <div key={index} className="result-item">
                    <p><strong>Emotion:</strong> {result.emotion}</p>
                    <p><strong>Confidence:</strong> {result.confidence}%</p>
                  </div>
                ))
              ) : (
                <p>No faces detected by FER model</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmotionDetector;