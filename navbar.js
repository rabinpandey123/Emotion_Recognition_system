import React from 'react';
import '../styles/navbar.css';

const Navbar = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h2>Emotion Detection</h2>
      </div>
      <div className="navbar-links">
        <button 
          className={activeTab === 'home' ? 'active' : ''} 
          onClick={() => setActiveTab('home')}
        >
          Home
        </button>
        <button 
          className={activeTab === 'detector' ? 'active' : ''} 
          onClick={() => setActiveTab('detector')}
        >
          Emotion Detector
        </button>
        <button 
          className={activeTab === 'contact' ? 'active' : ''} 
          onClick={() => setActiveTab('contact')}
        >
          Contact
        </button>
      </div>
    </nav>
  );
};

export default Navbar;