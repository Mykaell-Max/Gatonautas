import React from 'react';
import { Link } from 'react-router-dom';
import './Training.css';
import Training1 from '../../assets/Training1.svg';
import Training2 from '../../assets/Training2.svg';
import Training3 from '../../assets/Training3.svg';

const Training: React.FC = () => {
  return (
    <div className="training-container">
      {/* Título isolado com fundo preto, alinhado à esquerda */}
      <section className="section section-title">
        <div className="title-wrapper">
          <h1>Training</h1>
        </div>
      </section>

      {/* Seção 1 – imagem como fundo, texto sobreposto à direita */}
      <section
        className="section section-hero"
        style={{ backgroundImage: `url(${Training1})` }}
      >
        <div className="hero-overlay">
          <div className="hero-text">
            <h2>What are Exoplanets?</h2>
            <p>
              An exoplanet is any planet beyond our solar system. Most orbit other stars, but free-floating exoplanets, called rogue planets, orbit the galactic center and are untethered to any star.
            </p>
          </div>
        </div>
      </section>
      {/* Seção 3 – fundo branco com imagem lateral */}
      <section className="section section-side section-light">
        <div className="section-content">
          <h2>Why are exoplanets studied?</h2>
          <p>
            Studying exoplanets helps us understand the formation, evolution, and diversity of planetary systems. It can also provide insights into the potential habitability of other worlds and the possibility of life beyond Earth.
          </p>
          <p>
            By analyzing the atmospheres, compositions, and orbits of exoplanets, scientists can learn about the conditions that lead to planet formation and the factors that influence planetary environments.
          </p>
        </div>
        <div className="section-image">
          <img src={Training2} alt="Star field with transit" />
        </div>
      </section>

      {/* Seção 2 – fundo branco com texto explicativo */}
      <section className="section section-side section-light">
        <div className="section-content">
          <h2>How are exoplanets discovered?</h2>
          <p>
            A planetary transit occurs when a planet passes directly in front of its star, as viewed by an observer. During a transit, the planet blocks a small portion of the star’s light, causing a temporary and periodic dimming.
          </p>
          <p>
            This dimming can be detected by sensitive instruments and used to infer the presence of a planet. The amount of dimming and the duration of the transit can provide information about the planet’s size and orbit.
          </p>
          <p>
            The transit method is one of the most successful techniques for discovering exoplanets and is used by missions such as NASA’s Kepler and TESS.
          </p>
        </div>
<div className="section-video">
<iframe
  src="https://www.youtube.com/embed/xNeRqbw18Jk?autoplay=1&loop=1&playlist=xNeRqbw18Jk"
  title="Star field with transit"
  frameBorder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowFullScreen
></iframe>
</div>

      </section>


      {/* Seção 2 – fundo branco com texto explicativo */}
      <section className="section section-side section-light">
        <div className="section-content">
          <h2>Habitable Zone</h2>
          <p>
           One of the main requirements for the existence of life as we know is liquid water. The habitable zone is the range of distances from a star where liquid water could exist on the surface of an exoplanet. The temperature conditions there might be just right: not too hot and not too cold.
          </p>
        </div>
<div className="section-video">
<iframe
  src="https://upload.wikimedia.org/wikipedia/commons/transcoded/0/0e/Habitable_Zones_Compared_to_the_Size_of_the_Hosting_Star.webm/Habitable_Zones_Compared_to_the_Size_of_the_Hosting_Star.webm.720p.vp9.webm"
  title="Habitable Zones Compared to the Size of the Hosting Star"
  frameBorder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowFullScreen
></iframe>
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

export default Training;
