import React, { useState, useEffect, useRef } from 'react';
import { ChevronRight, ChevronLeft } from 'lucide-react';
import { fetchStockData } from '../services/api';
import { Line } from 'react-chartjs-2';
import LoadingSpinner from './LoadingSpinner';
import { getLogoUrl, StockLogoContainer } from '../utils/logoUtils';

const PortfolioSummary = () => {
  const [portfolioStocks, setPortfolioStocks] = useState([]);
  const [currentPage, setCurrentPage] = useState(0);
  const containerRef = useRef(null);
  const [itemsPerPage, setItemsPerPage] = useState(4);

  useEffect(() => {
    // Get optimization results from session storage
    const storedResults = JSON.parse(sessionStorage.getItem('optimizationResults') || '[]');
    if (storedResults.length > 0) {
      // Initialize portfolio stocks with optimization data
      const stocks = storedResults.map(stock => ({
        ...stock,
        data: null,
        loading: true
      }));
      setPortfolioStocks(stocks);

      // Fetch real-time data for each stock
      stocks.forEach(async (stock, index) => {
        try {
          const data = await fetchStockData(stock.symbol, '3mo');
          setPortfolioStocks(prev => {
            const updated = [...prev];
            updated[index] = {
              ...updated[index],
              data,
              loading: false
            };
            return updated;
          });
        } catch (error) {
          console.error(`Error fetching data for ${stock.symbol}:`, error);
        }
      });
    }
  }, []);

  useEffect(() => {
    const updateItemsPerPage = () => {
      if (containerRef.current) {
        const width = containerRef.current.offsetWidth;
        setItemsPerPage(Math.floor(width / 300));
      }
    };

    updateItemsPerPage();
    window.addEventListener('resize', updateItemsPerPage);
    return () => window.removeEventListener('resize', updateItemsPerPage);
  }, []);

  if (portfolioStocks.length === 0) {
    return (
      <div className="portfolio-summary">
        <h2>My Portfolio</h2>
        <p className="no-data">No previous optimizations found</p>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false
      }
    },
    scales: {
      x: {
        display: false,
        grid: {
          display: false
        }
      },
      y: {
        display: false,
        grid: {
          display: false
        },
        // Add min and max to "zoom in" on the data
        min: (context) => {
          const values = context.chart.data.datasets[0].data;
          const min = Math.min(...values);
          return min - (min * 0.001); // Slightly below minimum
        },
        max: (context) => {
          const values = context.chart.data.datasets[0].data;
          const max = Math.max(...values);
          return max + (max * 0.001); // Slightly above maximum
        }
      }
    },
    elements: {
      line: {
        tension: 0.4,
        borderWidth: 2
      },
      point: {
        radius: 0
      }
    },
    layout: {
      padding: {
        top: 5,
        bottom: 5
      }
    }
  };

  const getChartData = (stockData) => {
    if (!stockData?.data || !stockData.data.length) {
      console.log('No stock data available:', stockData);
      return null;
    }
  
    const isPositive = stockData.percentChange >= 0;
    const color = isPositive ? 'rgb(74, 222, 128)' : 'rgb(248, 113, 113)';
  
    // Get a reduced set of data points for smoother visualization
    const data = stockData.data;
    const step = Math.max(1, Math.floor(data.length / 50)); // Aim for about 50 points
    const reducedData = data.filter((_, index) => index % step === 0);
  
    return {
      labels: reducedData.map(item => item.date),
      datasets: [{
        label: 'Price',
        data: reducedData.map(item => item.close),
        borderColor: color,
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, context.chart.height);
          // Using rgba format for transparency instead of hex
          if (isPositive) {
            gradient.addColorStop(0, 'rgba(74, 222, 128, 0.2)');
            gradient.addColorStop(1, 'rgba(74, 222, 128, 0)');
          } else {
            gradient.addColorStop(0, 'rgba(248, 113, 113, 0.2)');
            gradient.addColorStop(1, 'rgba(248, 113, 113, 0)');
          }
          return gradient;
        },
        fill: true,
        borderWidth: 2,
        tension: 0.4,
        pointRadius: 0
      }]
    };
  };

  const getRemainingCards = () => {
    const visibleCards = (currentPage + 1) * 4;
    return Math.max(0, portfolioStocks.length - visibleCards);
  };
  
  const getAllStocks = () => {
    return portfolioStocks;
  };

  const nextPage = () => {
    const remainingCards = getRemainingCards();
    if (remainingCards > 0) {
      setCurrentPage(prev => prev + 1);
    }
  };
  const prevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(prev => prev - 1);
    }
  };

  return (
    <div className="portfolio-summary">
      <h2 className="portfolio-title">My Portfolio</h2>
      <div className="portfolio-container" ref={containerRef}>
        <div className="portfolio-cards">
          {currentPage > 0 && (
              <button onClick={prevPage} className="nav-button nav-button-left">
                <ChevronLeft className="nav-icon" />
              </button>
            )}
              <div 
                className="portfolio-cards-wrapper" 
                style={{ 
                  transform: `translateX(-${currentPage * (4 * 256)}px)`, // 240px card width + 16px gap
                  transition: 'transform 0.5s ease',
                  display: 'flex',
                  gap: '16px'
                }}
              >
            {getAllStocks().map((stock) => (
              <div key={stock.symbol} className="portfolio-card">
                <div className="card-header">
                  <div className="stock-logo-container">
                    <img
                      src={getLogoUrl(stock.symbol, stock.security)}
                      alt={stock.symbol}
                      className="stock-logo-summary"
                      onError={(e) => {
                        e.target.src = '/default-stock-logo.png';
                      }}
                    />
                    <span className="stock-name">{stock.symbol}</span>
                  </div>
                  <span className={`stock-percent-change ${stock.data?.percentChange >= 0 ? 'positive' : 'negative'}`}>
                    {stock.data?.percentChange >= 0 ? '+' : ''}
                    {stock.data?.percentChange?.toFixed(2)}%
                  </span>
                </div>
  
                  <div className="stock-chart">
                    {stock.data && stock.data.data && stock.data.data.length > 0 ? (
                      <Line
                        data={getChartData(stock.data)}
                        options={chartOptions}
                        height={50}
                      />
                    ) : (
                    <div className="loading-chart">Loading...</div>
                  )}
                </div>
  
                <div className="stock-price-value">
                  ${stock.data?.currentPrice?.toFixed(2) || '0.00'}
                </div>
              </div>
            ))}
            
          </div>
          {getRemainingCards() > 0 && (
            <button onClick={nextPage} className="nav-button nav-button-right">
              <ChevronRight className="nav-icon" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
  export default PortfolioSummary;