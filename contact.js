import React from 'react';
import '../styles/contact.css';

const Contact = () => {
  return (
    <div className="contact-container">
      <h1>Contact & Support</h1>
      
      <div className="contact-content">
        <div className="contact-info">
          <h2>Get Help</h2>
          <p>
            If you're experiencing issues with the emotion detection system, 
            here are some common solutions:
          </p>
          
          <div className="troubleshooting">
            <h3>Camera Not Working</h3>
            <ul>
              <li>Make sure your camera is connected and not being used by another application</li>
              <li>Try using the Camera Diagnostics tool in the Emotion Detector tab</li>
              <li>Check your browser's permission settings for camera access</li>
              <li>Try a different browser (Chrome or Firefox recommended)</li>
            </ul>
            
            <h3>Low Accuracy</h3>
            <ul>
              <li>Ensure good lighting on your face</li>
              <li>Look directly at the camera</li>
              <li>Remove glasses or hats that might obscure facial features</li>
              <li>Try the FER Model Diagnostics to see detailed confidence scores</li>
            </ul>
          </div>
        </div>
        
        <div className="support-form">
          <h2>Report an Issue</h2>
          <form>
            <div className="form-group">
              <label htmlFor="name">Name</label>
              <input type="text" id="name" placeholder="Your name" />
            </div>
            
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input type="email" id="email" placeholder="Your email" />
            </div>
            
            <div className="form-group">
              <label htmlFor="issue">Issue Description</label>
              <textarea 
                id="issue" 
                rows="5" 
                placeholder="Describe the problem you're experiencing..."
              ></textarea>
            </div>
            
            <button type="submit">Submit Report</button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Contact;