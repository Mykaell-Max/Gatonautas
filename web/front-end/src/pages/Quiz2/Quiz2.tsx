import React from 'react';
import { Link } from 'react-router-dom';
import './Quiz2.css';
import Classify1 from '../../assets/Classify1.svg';
import Training3 from '../../assets/Training3.svg';

const Quiz2: React.FC = () => {
  return (
    <div className="training-container">
      {/* Título isolado com fundo preto, alinhado à esquerda */}
      <section className="section section-title">
        <div className="title-wrapper">
          <h1>Classify</h1>
        </div>
      </section>

      {/* Seção 1 – imagem como fundo, texto sobreposto à direita */}
      <section
        className="section section-hero"
        style={{ backgroundImage: `url(${Classify1})` }}
      >
        <div className="hero-overlay">
          <div className="hero-text">
            <h2>Welcome to ExoLearn: Our Citizen Science Plataform</h2>
            <p>
              Join the search for new worlds!<br/>By analyzing real astronomical data, you’ll help identify planetary transits: the tiny dips in a star’s brightness caused by an exoplanet passing in front of it.<br/>Your classifications will train our machine learning models to detect exoplanets faster and more accurately.<br/>Every click brings us closer to finding another Earth!
            </p>
            <p>Our platform is in its prototype stage, so data collection hasn’t started yet, but you can already explore and test how it works!<br/>For now, feel free to try out the interface and be part of building the tools that will uncover the next Earth!</p>
          </div>
        </div>
      </section>

      {/* Seção 4 – imagem à direita, texto à esquerda em caixa cinza */}
      <section className="section section-side section-light reverse">
        <div className="section-content quiz-box">
          <h2>Ready to test your knowledge?</h2>
          <Link to="/quiz">
            <button className="quiz">Start Quiz</button>
          </Link>
        </div>

        <div className="section-content2">
          <h2>This is what a Planetary Transit looks like:</h2>
          <div className="section-image">
            <img src={Training3} alt="Transit graph" />
          </div>
        </div>
      </section>
    </div>
  );
};

export default Quiz2;
