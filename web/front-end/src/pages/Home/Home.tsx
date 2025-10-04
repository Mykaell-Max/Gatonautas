import HomeView from "./View/HomeView.tsx";
import AuthCard from "../../components/AuthCard.tsx";
import { useState } from "react";

function Home() {
  const [showAuth, setShowAuth] = useState(false);

  return (
    <div>
      {showAuth ? (
        <AuthCard />
      ) : (
        <HomeView onLoginClick={() => setShowAuth(true)} />
      )}
    </div>
  );
}

export default Home;
