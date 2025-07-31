import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { fetchStocks, fetchStockData, fetchSP500Details } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import PortfolioSummary from '../components/PortfolioSummary'; // Add this line
import { getLogoUrl, StockLogoContainer } from '../utils/logoUtils';

const getCompanyDomain = (companyName) => {
  if (!companyName) return '';
  return companyName
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')
    .replace(/^the/, '')
    .replace(/(inc|corp|ltd|llc)$/, '');
};

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend
);

const TimeRangeControl = ({ selectedRange, onChange }) => {
  const ranges = [
    { label: '1 Day', value: '1d' },
    { label: '5 Days', value: '5d' },
    { label: '1 Month', value: '1mo' },
    { label: '6 Months', value: '6mo' },
    { label: '1 Year', value: '1y' },
    { label: '5 Years', value: '5y' },
    { label: 'All', value: 'max' },
  ];

  return (
    <div className="time-range-controls">
      {ranges.map(range => (
        <button
          key={range.value}
          className={`time-range-btn ${selectedRange === range.value ? 'active' : ''}`}
          onClick={() => onChange(range.value)}
        >
          {range.label}
        </button>
      ))}
    </div>
  );
};

const Overview = () => {
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState(null);
  const [previousOptimizations, setPreviousOptimizations] = useState([]);
  const [sp500Data, setSP500Data] = useState(null);
  const [sp500Loading, setSP500Loading] = useState(true);
  const [timeRange, setTimeRange] = useState('1d');

  useEffect(() => {
    console.log("Selected Stock:", selectedStock?.value);
    console.log("Time Range:", timeRange);
  }, [selectedStock, timeRange]);
  
  useEffect(() => {
    const loadSP500Data = async () => {
      try {
        setSP500Loading(true);
        const data = await fetchSP500Details();
        setSP500Data(data);
      } catch (err) {
        setError('Failed to load S&P 500 data');
      } finally {
        setSP500Loading(false);
      }
    };

    loadSP500Data(); // Initial load
    const refreshInterval = setInterval(loadSP500Data, 10000); 

    return () => clearInterval(refreshInterval);
  }, []);

  useEffect(() => {
    const loadStocks = async () => {
      try {
        const data = await fetchStocks();
        setStocks(data);
        const storedOptimizations = sessionStorage.getItem('optimizationResults');
        if (storedOptimizations) {
          setPreviousOptimizations(JSON.parse(storedOptimizations));
        }
      } catch (err) {
        setError('Failed to load stocks');
      } finally {
        setLoading(false);
      }
    };

    loadStocks();
  }, []);

  useEffect(() => {
    const loadStockData = async () => {
      if (!selectedStock) return;
      
      setChartLoading(true);
      try {
        const data = await fetchStockData(selectedStock.value, timeRange);
        setStockData(data);
      } catch (err) {
        setError('Failed to load stock data');
      } finally {
        setChartLoading(false);
      }
    };

    loadStockData();
  }, [selectedStock, timeRange]);

  const getChartData = () => {
    if (!stockData) return null;

    return {
      labels: stockData.data.map(item => item.date),
      datasets: [{
        label: selectedStock?.value || 'Select a stock',
        data: stockData.data.map(item => item.close),
        borderColor: stockData.percentChange >= 0 ? 'rgb(74, 222, 128)' : 'rgb(248, 113, 113)',
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, context.chart.height);
          if (stockData.percentChange >= 0) {
            gradient.addColorStop(0, 'rgba(74, 222, 128, 0.2)'); // Green with opacity
            gradient.addColorStop(1, 'rgba(74, 222, 128, 0)');
          } else {
            gradient.addColorStop(0, 'rgba(248, 113, 113, 0.2)'); // Red with opacity
            gradient.addColorStop(1, 'rgba(248, 113, 113, 0)');
          }
          return gradient;
        },
        tension: 0.4,
        fill: true,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 0
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
        labels: {
          color: 'white',
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(27, 32, 40, 0.9)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: (context) => {
            return `$${context.parsed.y.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      y: {
        ticks: { 
          color: 'white',
          callback: (value) => `$${value.toFixed(2)}`
        },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      x: {
        ticks: { 
          /*color: 'white',
          maxTicksLimit: 8*/
          display: false
        },
        grid: {display: false
        } /*{ color: 'rgba(255, 255, 255, 0.1)' }*/
      }
    },
    interaction: {
      intersect: false,
      mode: 'index'
    },
    animation: {
      duration: 500, 
      easing: 'easeOutBounce' 
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="main-content">
      <div className="header">
        <h1>Dashboard</h1>
      </div>
      
      <PortfolioSummary />

      <div className="dashboard-container">
        <div className="main-section">
          <div className="dashboard-card stock-graph">
            <h2>Stock Performance</h2>
            
            <div className="stock-header">
              <Select
                options={stocks.map(stock => ({
                  value: stock.Symbol,
                  label: (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <img
                        src={getLogoUrl(stock.Symbol, stock.Security)}
                        alt={`${stock.Security} logo`}
                        style={{
                          width: '24px',
                          height: '24px',
                          borderRadius: '50%',
                          backgroundColor: 'white',
                          padding: '2px'
                        }}
                        onError={(e) => e.target.src = '/default-stock-logo.png'}
                      />
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontWeight: '500' }}>{stock.Symbol}</span>
                        <span style={{ fontSize: '0.8em', color: 'rgba(255, 255, 255, 0.7)' }}>
                          {stock.Security}
                        </span>
                      </div>
                    </div>
                  )
                }))}
                styles={{
                  control: (provided) => ({
                    ...provided,
                    backgroundColor: 'rgba(49, 53, 63, 1)',
                    borderColor: 'rgba(128, 128, 128, 1)',
                    color: 'white',
                    minHeight: '50px',
                    '&:hover': {
                      borderColor: 'rgba(57, 110, 247, 1)'
                    }
                  }),
                  menu: (provided) => ({
                    ...provided,
                    backgroundColor: 'rgba(27, 32, 40, 1)',
                    padding: '8px',
                    border: '1px solid rgba(57, 110, 247, 0.3)',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                  }),
                  option: (provided, state) => ({
                    ...provided,
                    backgroundColor: state.isFocused ? 'rgba(57, 110, 247, 0.1)' : 'transparent',
                    color: 'white',
                    padding: '12px',
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: 'rgba(57, 110, 247, 0.2)',
                    }
                  }),
                  singleValue: (provided) => ({
                    ...provided,
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center'
                  }),
                  input: (provided) => ({
                    ...provided,
                    color: 'white'
                  }),
                  placeholder: (provided) => ({
                    ...provided,
                    color: 'rgba(255, 255, 255, 0.5)'
                  }),
                  menuList: (provided) => ({
                    ...provided,
                    '::-webkit-scrollbar': {
                      width: '8px'
                    },
                    '::-webkit-scrollbar-track': {
                      background: 'rgba(27, 32, 40, 1)',
                      borderRadius: '4px'
                    },
                    '::-webkit-scrollbar-thumb': {
                      background: 'rgba(57, 110, 247, 0.5)',
                      borderRadius: '4px',
                      '&:hover': {
                        background: 'rgba(57, 110, 247, 0.7)'
                      }
                    }
                  })
                }}
                onChange={(selectedOption) => {
                  if (!selectedOption) {
                    setSelectedStock(null);
                    return;
                  }
                  // Find the full stock info from our stocks array
                  const stockInfo = stocks.find(s => s.Symbol === selectedOption.value);
                  setSelectedStock({
                    value: selectedOption.value,
                    label: selectedOption.value,
                    security: stockInfo.Security // Store the full company name
                  });
                }}

                placeholder="Select a stock..."
                className="stock-select"
                isSearchable
                isClearable
              />
              
              {selectedStock && stockData && (
                <div className="selected-stock-info">
                  <div className="stock-identity">
                    <div className="stock-logo">
                      <img 
                        src={`https://logo.clearbit.com/${getCompanyDomain(selectedStock.security)}.com`}
                        alt={selectedStock.value}
                        onError={(e) => e.target.src = '/default-stock-logo.png'}
                      />
                    </div>
                    <div className="stock-name-container">
                      <span className="stock-symbol">{selectedStock.value}</span>
                      <span className="timestamp">
                        {new Date().toLocaleString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                          timeZoneName: 'short'
                        })}
                        · INDEXSP · Disclaimer
                      </span>
                    </div>
                  </div>
                  <div className="stock-price-info">
                    <span className="current-price">${stockData.currentPrice.toFixed(2)}</span>
                    <span className={`price-change ${stockData.percentChange >= 0 ? 'positive' : 'negative'}`}>
                      {stockData.percentChange >= 0 ? '+' : ''}
                      {stockData.percentChange.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="graph-container">
              {chartLoading ? (
                <div className="chart-loading">
                  <LoadingSpinner />
                </div>
              ) : stockData ? (
                <Line data={getChartData()} options={chartOptions} />
              ) : (
                <div className="no-data">Select a stock to view its performance</div>
              )}
            </div>

            <div className="graph-footer">
              <TimeRangeControl
                selectedRange={timeRange}
                onChange={setTimeRange}
              />
            </div>
          </div>
        </div>
          {/* S&P 500 Details Container */}
          <div className="side-section">
            <div className="dashboard-card sp500-details">
              <div className="section-header">
                <h2>S&P 500 Details</h2>
                <span className="time-indicator">• Live</span>
              </div>
              {sp500Loading ? (
                <div className="loading-container">
                  <LoadingSpinner />
                </div>
              ) : sp500Data ? (
                <div className="details-grid">
                  <div className="detail-item">
                    <span className="label">Current Price</span>
                    <span className="value">
                      ${sp500Data.currentPrice.toFixed(2)}
                      <span className={`change ${sp500Data.percentChange >= 0 ? 'positive' : 'negative'}`}>
                        ({sp500Data.percentChange >= 0 ? '+' : ''}
                          {sp500Data.percentChange ? sp500Data.percentChange.toFixed(2) : '0.00'}%)
                      </span>
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Previous Close</span>
                    <span className="value">${sp500Data.previousClose.toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Day Range</span>
                    <span className="value">
                      ${sp500Data.dayRange.low.toFixed(2)} - ${sp500Data.dayRange.high.toFixed(2)}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Year Range</span>
                    <span className="value">
                      ${sp500Data.yearRange.low.toFixed(2)} - ${sp500Data.yearRange.high.toFixed(2)}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Market Cap</span>
                    <span className="value">
                      ${sp500Data.marketCap.toFixed(2)}T
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Volume</span>
                    <span className="value">
                      {new Intl.NumberFormat().format(sp500Data.volume)}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="error-message">Failed to load S&P 500 data</div>
              )}
            </div>

            {/* Previous Optimizations Container */}
            <div className="dashboard-card previous-optimizations">
              <h2>Previous Optimizations</h2>
              {previousOptimizations.length > 0 ? (
                <div className="optimization-list">
                  {previousOptimizations.map((item) => (
                    <div key={item.symbol} className="optimization-item">
                      <div className="optimization-header">
                        <span className="symbol">{item.symbol}</span>
                      </div>
                      <span class="symbol"></span>
                      <div className="optimization-details">
                        <div className="detail">
                          <span className="label">Weight</span>
                          <span className="value">{(item.weight * 100).toFixed(2)}%</span>
                        </div>
                        <div className="detail">
                          <span className="label">Expected Return</span>
                          <span className="value">{(item.expected_return * 100).toFixed(2)}%</span>
                        </div>
                        <div className="detail">
                          <span className="label">Balance</span>
                          <span className="value">${item.allocation.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-data">No previous optimizations found</p>
              )}
            </div>
          </div>
      </div>
    </div>
  );
};

export default Overview;