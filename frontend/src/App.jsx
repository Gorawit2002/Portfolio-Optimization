import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Overview from './pages/Overview';
import Optimize from './pages/Optimize';
import OptimizeStocks from './pages/OptimizeStocks';
import OptimizeResults from './pages/OptimizeResults';
import ErrorBoundary from './components/ErrorBoundary';

const App = () => {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/optimize" element={<Optimize />} />
            <Route path="/optimize/stocks" element={<OptimizeStocks />} />
            <Route path="/optimize/results" element={<OptimizeResults />} />
          </Routes>
        </ErrorBoundary>
      </div>
    </Router>
  );
};

export default App;