import { useState } from "react";
import "./AuthCard.css";

export default function AuthCard() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [isOpen, setIsOpen] = useState(true); // controle de exibição do modal

  const toggleAuth = () => setIsSignUp(!isSignUp);
  const closeModal = () => setIsOpen(false);

  if (!isOpen) return null; // não renderiza nada se modal fechado

  return (
    <div className="auth-overlay">
      <div className={`auth-modal ${isSignUp ? "flip" : ""}`}>
        {/* Botão de fechar */}
        <span className="auth-close-btn" onClick={closeModal}>
          ×
        </span>

        {/* Front: Login */}
        <div className="auth-front">
          <h2>Log In</h2>
          <div className="auth-form">
            <div className="form-group">
              <label>Email</label>
              <input type="email" placeholder="Digite seu email" />
              <span className="input-icon">✉️</span>
            </div>
            <div className="form-group">
              <label>Password</label>
              <input type="password" placeholder="Digite sua senha" />
              <span className="input-icon">🔒</span>
            </div>
            <button className="auth-submit-btn">Log In</button>
          </div>
          <p className="auth-toggle-text">
            Don't have an account?{" "}
            <span className="auth-toggle-link" onClick={toggleAuth}>
              Sign Up
            </span>
          </p>
        </div>

        {/* Back: Sign Up */}
        <div className="auth-back">
          <h2>Sign Up</h2>
          <div className="auth-form">
            <div className="form-group">
              <label>Name</label>
              <input type="text" placeholder="Digite seu nome" />
              <span className="input-icon">👤</span>
            </div>
            <div className="form-group">
              <label>Email</label>
              <input type="email" placeholder="Digite seu email" />
              <span className="input-icon">✉️</span>
            </div>
            <div className="form-group">
              <label>Password</label>
              <input type="password" placeholder="Digite sua senha" />
              <span className="input-icon">🔒</span>
            </div>
            <button className="auth-submit-btn">Sign Up</button>
          </div>
          <p className="auth-toggle-text">
            Already have an account?{" "}
            <span className="auth-toggle-link" onClick={toggleAuth}>
              Log In
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
