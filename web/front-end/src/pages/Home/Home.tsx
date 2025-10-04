import React, { useState } from "react";
import "./Home.css";
import AuthCard from "../../components/AuthCard";

interface HomeProps {
  onLoginClick: () => void;
}

const Home: React.FC<HomeProps> = ({ onLoginClick }) => {
  const [showAuthCard, setShowAuthCard] = useState(false);

  const handleLoginClick = () => {
    setShowAuthCard(true);
  };

  const handleCloseAuthCard = () => {
    setShowAuthCard(false);
  };

  return (
    <>
      <div className="home-container">
        {/* Navbar */}
        <nav className="navbar">
          <span className="logo">Gatonautas Org.</span>
          <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">Learn</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#">Contact</a></li>
            <li>
              <button 
                onClick={handleLoginClick} 
                className="hover:text-purple-400 font-bold"
              >
                Log In
              </button>
            </li>
          </ul>
        </nav>

        {/* Hero Section */}
        <section className="hero">
          <div className="hero-text">
            <h1>
              A World Away: <br />
              <span className="highlight">Exoplanet Hunting</span>
            </h1>
            <p>
              Explore new worlds. Learn how to bring Space Science to Earth.
            </p>
            <div className="hero-buttons">
              <button className="explore">Explore</button>
              <button className="learn">Learn</button>
            </div>
          </div>

          <div className="hero-description">
            <p>
              Explore new worlds. Harness the power of AI to uncover hidden planets outside the Solar System.
            </p>
            <p>
              Our project combines machine learning, education, and citizen science to make exoplanet discovery accessible to everyone from researchers analyzing their own data to students taking their first steps in astronomy.
            </p>
            <p>Upload real light-curve data, experiment with AI models, and help identify planetary transits in telescope images. Together, we're building a bridge between curiosity and discovery, turning data into the stories of distant worlds.</p>
          </div>
        </section>
      </div>

      {/* Modal do AuthCard - FORA do home-container para ficar por cima */}
      {showAuthCard && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999
          }}
          onClick={handleCloseAuthCard}
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            style={{ position: 'relative' }}
          >
            <button
              onClick={handleCloseAuthCard}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'none',
                border: 'none',
                color: 'white',
                fontSize: '32px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              ✕
            </button>
            <AuthCard />
          </div>
        </div>
      )}
    </>
  );
};

export default Home;