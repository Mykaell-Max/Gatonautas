import React from "react";
import "./Home.css";

const Home: React.FC = () => {
  return (
    <div className="home-container">
      {/* Navbar */}
      <nav className="navbar">
        <span className="logo">Gatonautas Org.</span>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">Learn</a></li>
          <li><a href="#">About</a></li>
          <li><a href="#">Contact</a></li>
          <li><a href="#" className="login">Log In</a></li>
        </ul>
      </nav>

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
            <button className="learn">Learn</button>
          </div>
        </div>

        <div className="hero-description">
          <p>
            Founded by a group of dedicated students, the Data Science Club brings together individuals from diverse academic backgrounds who share a common interest in data science, machine learning, artificial intelligence, and related technologies.
          </p>
          <p>
            Our members range from beginners to advanced practitioners, all united by a desire to explore the fascinating world of data.
          </p>
        </div>
      </section>
    </div>
  );
};

export default Home;
