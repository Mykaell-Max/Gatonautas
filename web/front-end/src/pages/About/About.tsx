import React from "react";
import "./About.css";
import GalacticCat from "../../assets/galactic-cat.svg"; // substitua pelo caminho da sua imagem

const About: React.FC = () => {
  return (
    <div className="about-container"
                style={{
        backgroundImage: `url(${GalacticCat})`,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'right top',
        backgroundSize: '50%'
      }} >
      <div className="about-text">
        <h1>About.</h1>
        <p>
          We are a team of students who are passionate about technology and science dissemination. Machine learning applied to astronomical research can be challenging and inaccessible. Therefore, we aim to bring exoplanet discovery to a simple, beginner-friendly interface while also offering advanced options for researchers. In addition, we propose tools focused on both theoretical and practical learning. Thus, we seek to bring science to people, so that we can, in turn, bring people to science.
        </p>
      </div>
    </div>
  );
};

export default About;
