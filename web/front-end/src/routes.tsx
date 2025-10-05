import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Home from "./pages/Home/Home";
import Learn from "./pages/Learn/Learn";
import Training from "./pages/Learn/Training/Training"
import Explore from "./pages/Explore/Explore";

export const AppRoutes = () =>{
    return(
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Learn" element={<Learn />} />
                <Route path="/Training" element={<Training />} />
                <Route path="/Explore" element={<Explore/>} />

            </Routes>
    )
}

