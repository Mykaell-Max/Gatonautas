import { useState } from "react";
import AuthCard from "../AuthCard"; // ajuste o caminho conforme sua estrutura
import "./Navbar.css"

const Navbar: React.FC = () => {
  const [showAuthCard, setShowAuthCard] = useState(false);

  const handleLoginClick = () => setShowAuthCard(true);
  const handleCloseAuthCard = () => setShowAuthCard(false);

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

      {showAuthCard && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.7)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 9999,
          }}
          onClick={handleCloseAuthCard} // fecha clicando no overlay
        >
          <div onClick={(e) => e.stopPropagation()} style={{ position: "relative" }}>
            {/* Botão de fechar */}
            <button
              onClick={handleCloseAuthCard}
              style={{
                position: "absolute",
                top: "-40px",
                right: "0",
                background: "none",
                border: "none",
                color: "white",
                fontSize: "32px",
                cursor: "pointer",
                fontWeight: "bold",
              }}
            >
              ✕
            </button>

            {/* Componente de login/signup */}
            <AuthCard />
          </div>
        </div>
      )}
    </>
  );
};

export default Navbar;
