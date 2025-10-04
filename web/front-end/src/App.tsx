import { useState } from "react";
import Home from "./pages/Home/Home.tsx";
import AuthCard from "./components/AuthCard.tsx";

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
