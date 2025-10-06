import { useState } from "react";
import { Link } from "react-router-dom";
import AuthCard from "../AuthCard";
import "./Navbar.css";

const Navbar: React.FC = () => {
  const [showAuthCard, setShowAuthCard] = useState(false);

  const handleLoginClick = () => setShowAuthCard(true);
  const handleClose = () => setShowAuthCard(false);

  return (
    <>
      <nav className="navbar">
        <Link to="/"><span className="logo">Gatonautas Org.</span></Link>
        <ul>
          <li><Link to="/Learn">Learn</Link></li>
          <li><Link to="/Explore">Explore</Link></li>
          <li><Link to="/About">About</Link></li>
        </ul>
      </nav>

      {showAuthCard && <AuthCard onClose={handleClose} />}
    </>
  );
};

export default Navbar;
