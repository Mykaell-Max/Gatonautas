import { AppRoutes } from "./routes";
import Navbar from "./components/Navbar/Navbar";
import AuthCard from "./components/AuthCard";

function App() {
  return (
    <>
      <Navbar />
      <AppRoutes />
    </>
  );
}

export default App;
