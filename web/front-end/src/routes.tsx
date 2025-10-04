import {BrowserRouter as Router, Routes, Route} from "react-router-dom";
import Home from "./pages/Home/Home";
import Learn from "./pages/Learn/Learn";

export const AppRoutes = () =>{
    return(
        <Router>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/Learn" element={<Learn />} />

            </Routes>
        </Router>
    )
}

