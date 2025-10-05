import React from "react";
import "./Learn.css";
import { Link } from "react-router-dom";
import Foguete from "../../assets/Learn.svg";

const LearnView: React.FC = () => {
  return (
    <div 
      className="learn-container"
      style={{
        backgroundImage: `url(${Foguete})`,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'right top',
        backgroundSize: '35%'
      }}
    >
      <div className="learn-hero">
        {/* Bloco de texto */}
        <div className="learn-text">
          <h1>Learn.</h1>
         
            {/* Coluna da esquerda */}
           
              
                <p>
                  From the fundamentals of planetary transits to Machine Learning tools, dive into interactive lessons and challenges.
                </p>
                <p>
                  Empower your curiosity and become part of the next generation of explorers.
                </p>
                <div className="learn-buttons">
                <Link to="/Training">
                  <button className="training">Training</button>
                </Link>
              </div>
           

            {/* Coluna da direita */}
            <div className="learn-right">
                <p>Ready to go? Classify real data and help new discoveries!</p>
                <div className="learn-buttons">
                <Link to="/Quiz2">
                  <button className="Quiz2">Classify</button>
                </Link>
              </div>
            </div>
          
        </div>
      </div>

     
    </div>
  );
};

export default LearnView;