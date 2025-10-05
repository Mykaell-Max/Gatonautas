import "../Home.css";
import { Link } from "react-router-dom";
import PlanetImage from "../../../assets/Home.svg";

const HomeView = () => {
  return (
    <div className="home-container">
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
            <Link to="/Learn">
              <button className="learn">Learn</button>
            </Link>
          </div>
        </div>

        <div className="hero-description">
          <p>
            Explore new worlds. Harness the power of AI to uncover hidden planets outside the Solar System.
          </p>
          <p>
            Our project combines machine learning, education, and citizen science to make exoplanet discovery accessible to everyone—from researchers analyzing their own data to students taking their first steps in astronomy.
          </p>
          <p>
            Upload real light-curve data, experiment with AI models, and help identify planetary transits in telescope images. Together, we're building a bridge between curiosity and discovery, turning data into the stories of distant worlds.
          </p>
        </div>
      </section>

      {/* Planet Image */}
      <img src={PlanetImage} alt="Planet illustration" className="planet-image" />
    </div>
  );
};

export default HomeView;
