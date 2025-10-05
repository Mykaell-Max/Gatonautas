import React from "react";
import "./Explore.css";
import { Link } from "react-router-dom";
import PlanetImage from "../../assets/2Planetas.svg";

const ExploreView: React.FC = () => {
  return (
    <div className="learn-container"
                style={{
        backgroundImage: `url(${PlanetImage})`,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'right top',
        backgroundSize: '100%'
      }}>
      <div className="learn-hero">
        <div className="learn-text">
          {/* Primeiro bloco: Upload + explicação */}
          <h1>Explore New Worlds.</h1>
          <p>
Find New Worlds. Upload your own light curves or transit data files to test and train the model.
Our system automatically processes your input, retrains the AI, and updates performance statistics,
helping you uncover potential exoplanet signals in minutes. You can also adjust hyperparameters directly from the interface to see how these changes impact your model’s results.
          </p>
          <div className="learn-buttons">
            <Link to="/UploadData">
              <button className="training">Upload Data & Adjust Parameters</button>
            </Link>
          </div>

          {/* Segundo bloco: Métricas e especificações */}
            <p>
Explore detailed specifications of the model and performance metrics across tested datasets.
Understand how the AI evaluates your data, and visualize results to improve your scientific insights.
            </p>
             <div className="learn-buttons">
            <button className="classify">View Model Metrics</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExploreView;
