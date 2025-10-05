import "./ExoLearn.css";
import { Link } from "react-router-dom";
import PlanetImage from "../../../src/assets/Classify1.svg";

const ExoLearn = () => {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-text">
          <h1>
            Welcome to ExoLearn: <br />
            <span className="highlight">Our Citizen Science Plataform</span>
          </h1>
          <p>
            Join a global mission to uncover new worlds—and help bring the wonders of space science into everyday life.
          </p>
          <div  className="hero-buttons" >
            <Link to="/Explore">
            <button className="explore" >Start!!</button>
            </Link>
          </div>
        </div>

        <div className="hero-description">
          <p>
            Join the search for new worlds!<br/>By analyzing real astronomical data, you’ll help identify planetary transits: the tiny dips in a star’s brightness caused by an exoplanet passing in front of it.Your classifications will train our machine learning models to detect exoplanets faster and more accurately.<br/>Every click brings us closer to finding another Earth!
          </p>
          <p>
            The platform is in its prototype stage, so data collection hasn’t started yet, but you can already explore and test how it works!<br/>For now, feel free to try out the interface and be part of building the tools that will uncover the next Earth!
          </p>
        </div>
      </section>

      {/* Planet Image */}
      <img src={PlanetImage} alt="Planet illustration" className="planet-image" />
    </div>
  );
};

export default ExoLearn;
