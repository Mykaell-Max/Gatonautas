import { useState } from "react";
import "./AuthCard.css";

export default function AuthCard({ onClose }: { onClose: () => void }) {
  const [isSignUp, setIsSignUp] = useState(false);

  const toggleAuth = () => setIsSignUp(!isSignUp);

  return (
    <div className="auth-overlay">
      <div className={`auth-modal ${isSignUp ? "flip" : ""}`}>
        {/* Botão de fechar */}
        <span className="auth-close-btn" onClick={onClose}>
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
