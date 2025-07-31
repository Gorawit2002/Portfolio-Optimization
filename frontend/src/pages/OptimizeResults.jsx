// src/pages/OptimizeResults.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { formatCurrency, formatPercentage } from '../utils/helpers';
import LoadingSpinner from '../components/LoadingSpinner';
import ParentSize from '@visx/responsive/lib/components/ParentSize';
import PortfolioPieChart from '../components/AnimatedPieChart';
import { getLogoUrl } from '../utils/logoUtils';  // Add this line

const OptimizeResults = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const storedResults = JSON.parse(sessionStorage.getItem('optimizationResults'));
    console.log('Retrieved from sessionStorage:', storedResults);
  
    if (!storedResults) {
      console.error('No results found in sessionStorage');
      navigate('/optimize');
      return;
    }
  
    if (storedResults.allocations && Array.isArray(storedResults.allocations)) {
      setResults(storedResults.allocations);
    } else if (Array.isArray(storedResults)) {
      setResults(storedResults);
    } else {
      console.error('Invalid results format:', storedResults);
      navigate('/optimize');
    }
  }, [navigate]);
  
  if (!results) return <LoadingSpinner />;

  
  return (
    <div className="main-content">
      <div className="header">
        <h1>My Portfolio</h1>
      </div>

      <div className="results-container">
      <div className="chart-container">
        <ParentSize>
          {({ width, height }) => (
            <PortfolioPieChart
              width={width}
              height={height}
              data={results}
            />
          )}
        </ParentSize>
      </div>
        <div className="portfolio-table">
          <table>
            <thead>
              <tr>
                <th style={{ width: '200px' }}>Symbol</th>
                <th>Expected Return (%)</th>
                <th>Volatility (%)</th>
                <th>Sharpe Ratio</th>
                <th>Weight (%)</th>
                <th>Balance ($)</th>
              </tr>
            </thead>
            <tbody>
              {results.map((item) => (
                <tr key={item.symbol}>
                  <td>
                    <div className="stock-cell">
                    <img
                        src={getLogoUrl(item.symbol, item.security)}
                        alt={`${item.symbol} logo`}
                        className="stock-logo-table"
                        onError={(e) => e.target.src = '/default-stock-logo.png'}
                      />
                      <span>{item.symbol}</span>
                    </div>
                  </td>
                  <td>{(item.expected_return * 100).toFixed(2)}</td>
                  <td>{(item.volatility * 100).toFixed(2)}</td>
                  <td>{item.sharpe_ratio.toFixed(2)}</td>
                  <td>{(item.weight * 100).toFixed(2)}</td>
                  <td>{formatCurrency(item.allocation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="optimize-button">
          <button onClick={() => navigate('/')}>Done</button>
        </div>
      </div>
    </div>
  );
};

export default OptimizeResults;