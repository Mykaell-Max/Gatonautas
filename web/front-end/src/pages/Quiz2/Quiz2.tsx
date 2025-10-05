import React from 'react';
import { Link } from 'react-router-dom';
import './Quiz2.css';
import Training3 from '../../assets/Training3.svg';

const Quiz2: React.FC = () => {
  return (
    <div className="training-container">
      {/* Título isolado com fundo preto, alinhado à esquerda */}
      <section className="section section-title">
        <div className="title-wrapper">
          <h1>Training</h1>
        </div>
      </section>

      {/* Seção 4 – imagem à direita, texto à esquerda em caixa cinza */}
      <section className="section section-side section-light reverse">
        <div className="section-content Classify-box"  style={{ border: '2px solid #fff' }}>
          <h2>Your practice area — try detecting planetary transits yourself!</h2>
          <Link to="/Classify">
            <button  style={{ border: '2px solid #fff' }} className="Classify">GO!</button>
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
