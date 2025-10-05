import { useState } from "react";
import { Link } from "react-router-dom";
import AuthCard from "../AuthCard";
import "./Navbar.css";

const Navbar: React.FC = () => {
  const [showAuthCard, setShowAuthCard] = useState(false);

  const handleLoginClick = () => setShowAuthCard(true);

  return (
    <>
      <nav className="navbar">
        <Link to="/"><span className="logo">Gatonautas Org.</span></Link>
        <ul>
          <li><Link to="/Learn">Learn</Link></li>
          <li><Link to="/Explore">Explore</Link></li>
          <li><Link to="/contact">About</Link></li>
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
