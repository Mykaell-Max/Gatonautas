import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Home from "./pages/Home/Home";
import Learn from "./pages/Learn/Learn";
import Training from "./pages/Training/Training"
import Explore from "./pages/Explore/Explore";
import About from "./pages/About/About";
import UploadData from "./pages/Explore/UploadData/UploadData";
import Quiz from "./pages/Quiz/quiz";
import Classify from "./pages/Classify/Classify"
import Quiz2 from "./pages/Quiz2/Quiz2"

export const AppRoutes = () =>{
    return(
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Learn" element={<Learn />} />
                <Route path="/Training" element={<Training />} />
                <Route path="/Explore" element={<Explore/>} />
                <Route path="/About" element={<About/>} />
                <Route path="/UploadData" element={<UploadData/>} />
                <Route path="/Quiz" element={<Quiz />} />
                <Route path="/Classify" element={<Classify/>} />
                <Route path="/Quiz2" element={<Quiz2/>} />

            </Routes>
    )
}

