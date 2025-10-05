// src/pages/Training/Training.tsx

import React from 'react';
import './Training.css';
import transitImage from '../../assets/transit-example.jpg'; // substitua com o caminho correto da imagem

const Training: React.FC = () => {
  return (
    <div className="learn-container">
      <div className="learn-hero">
        <div className="learn-text">
          <h1>Training</h1>
          <p>
            <strong>What are Exoplanets?</strong><br /><br />
            An exoplanet is any planet beyond our solar system. They can orbit other stars, besides the sun, pulsars or even float free (rogue planets) untethered to any star.
            <br /><br />
          </p>
          <p>
            <strong>How do we discover exoplanets?</strong><br /><br />
            A planetary transit happens when a planet passes directly in front of its star, from a point of view, blocking a tiny portion of the star’s light for a short time. Imagine watching a bright light bulb and a small toy moving in front of it for a moment—the bulb looks slightly dimmer. That’s exactly what astronomers detect when an exoplanet transits its star.
            <br /><br />
            During a transit, the brightness of the star dips slightly, and by measuring this dip, scientists can learn a lot about the planet such as its size, orbital period, and even whether it has an atmosphere or the star’s influence filtering through it.
            <br /><br />
            This method is incredibly powerful because it’s sensitive and efficient. Space telescopes like Kepler and TESS monitor thousands of stars at once, looking for these tiny, regular dips in brightness. As a result, most of the exoplanets we know today have been discovered through the transit method.
          </p>
          <p>
            <strong>This is what a Planetary Transit looks like:</strong>
          </p>
          <div className="learn-buttons">
            <button className="quiz">Start Quiz</button>
          </div>
        </div>
        <div className="learn-image">
          <img src={transitImage} alt="Planetary transit illustration" />
        </div>
      </div>
    </div>
  );
};

export default Training;
