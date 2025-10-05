import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Home from "./pages/Home/Home";
import Learn from "./pages/Learn/Learn";
import Training from "./pages/Learn/Training/Training"

export const AppRoutes = () =>{
    return(
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Learn" element={<Learn />} />
                <Route path="/Training" element={<Training />} />

            </Routes>
    )
}

