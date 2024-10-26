import React, { useEffect, useMemo, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";

// Import pages
import Login from "./pages/Login";
import Signup from "./pages/SignUp";
import Build from "./pages/Build";
import Train from "./pages/Train";
import Validate from "./pages/Validate";
import Deploy from "./pages/Deploy";
import Data from "./pages/Data";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";

const App = () => {
  const [init, setInit] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => {
      setInit(true);
    });
  }, []);

  const particlesLoaded = (container) => {
    console.log(container);
  };

  const options = useMemo(
    () => ({
      background: {
        color: { value: "#FFFFFF" },
      },
      fpsLimit: 120,
      interactivity: {
        events: {
          onClick: { enable: true, mode: "push" },
        },
        modes: {
          push: { quantity: 4 },
          repulse: { distance: 200, duration: 0.4 },
        },
      },
      particles: {
        color: { value: "#000000" },
        links: { color: "#000000", distance: 150, enable: true, opacity: 0.5, width: 1 },
        move: { enable: true, speed: 3, outModes: { default: "bounce" } },
        number: { density: { enable: true }, value: 80 },
        opacity: { value: 0.5 },
        shape: { type: "circle" },
        size: { value: { min: 1, max: 5 } },
      },
      detectRetina: true,
    }),
    []
  );

  return (
    <Router>
        {/* Navigation */}

        {/* Particles Background */}
        {/*init && <Particles id="tsparticles" particlesLoaded={particlesLoaded} options={options} style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              zIndex: -1, // Ensures particles are behind everything
            }}/>*/}

        {/* Page Routes */}
        <Routes> 
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/build" element={<Build />} />
          <Route path="/train" element={<Train />} />
          <Route path="/validate" element={<Validate />} />
          <Route path="/deploy" element={<Deploy />} />
          <Route path="/data" element={<Data />} />
        </Routes>
    </Router>
  );
};

export default App;
