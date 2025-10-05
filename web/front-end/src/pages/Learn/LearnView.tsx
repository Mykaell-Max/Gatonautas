// src/pages/Learn/LearnView.tsx
import React from "react";
import "./Learn.css"; // seu CSS para a página Learn

const LearnView: React.FC = () => {
  return (
    <>
      <div className="learn-container">
        <div className="learn-hero">
          <div className="learn-text">
            <h1>Learn.</h1>
            <p>
              From the fundamentals of planetary transits to Machine Learning tools, dive into interactive lessons and challenges.
            </p>
            <p>Empower your curiosity and become part of the next generation of explorers.</p>
            <div className="learn-buttons">
              <button className="training">Training</button>
            </div>
            <div className="learn-buttons">
            <p>Ready to go? Classify real data and help new discoveries!</p>  
              <button className="classify">Classify</button>
            </div>
          </div>

          <div className="learn-image">
            <img src="/images/rocket-launch.jpg" alt="Rocket Launch" />
          </div>
        </div>
      </div>
    </>
  );
};

export default LearnView;
