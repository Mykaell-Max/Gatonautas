import { useState } from "react";
import AuthCard from "../AuthCard"; 
import "./Navbar.css";

const Navbar: React.FC = () => {
  const [showAuthCard, setShowAuthCard] = useState(false);

  const handleLoginClick = () => setShowAuthCard(true);

  return (
    <>
      <nav className="navbar">
        <span className="logo">Gatonautas Org.</span>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">Learn</a></li>
          <li><a href="#">About</a></li>
          <li><a href="#">Contact</a></li>
          <li>
            <button onClick={handleLoginClick}>Log In</button>
          </li>
        </ul>
      </nav>

      {showAuthCard && <AuthCard />}
    </>
  );
};

export default Navbar;
