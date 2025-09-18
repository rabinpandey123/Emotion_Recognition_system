import React, { useState } from 'react';

// Component imports
import Navbar from './components/navbar';
import Home from './components/home';
import EmotionDetector from './components/Emotiondetector';
import Contact from './components/contact';

// CSS imports (make sure file names match exactly)
// import './App.css';
// import './components/Navbar.css';
// import './components/Home.css';
// import './components/EmotionDetector.css';
// import './components/Contact.css';

function App() {
  const [activeTab, setActiveTab] = useState('home');

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <Home />;
      case 'detector':
        return <EmotionDetector />;
      case 'contact':
        return <Contact />;
      default:
        return <Home />;
    }
  };

  return (
    <div className="App">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
