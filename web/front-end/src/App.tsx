import { useState } from "react";
import Home from "./Home";
import AuthCard from "./components/AuthCard";

function App() {
  const [showAuth, setShowAuth] = useState(false);

  return (
    <div>
      {showAuth ? (
        <AuthCard />
      ) : (
        <Home onLoginClick={() => setShowAuth(true)} />
      )}
    </div>
  );
}

export default App;
