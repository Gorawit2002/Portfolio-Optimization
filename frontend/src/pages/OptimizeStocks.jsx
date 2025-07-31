import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchStocks, optimizePortfolio } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';
import Select from 'react-select';

const OptimizeStocks = () => {
  const navigate = useNavigate();
  const [stocks, setStocks] = useState([]);
  const [selectedStocks, setSelectedStocks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const optimizationParams = JSON.parse(sessionStorage.getItem('optimizationParams') || '{}');

  useEffect(() => {
    const loadStocks = async () => {
      try {
        setLoading(true);
        const stockList = await fetchStocks();
        const stocksWithLogos = stockList.map(stock => ({
          ...stock,
          logoUrl: `https://logo.clearbit.com/${getCompanyDomain(stock.Security)}`
        }));
        setStocks(stocksWithLogos);
      } catch (err) {
        setError('Failed to load stocks');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadStocks();
  }, []);

  const getCompanyDomain = (companyName) => {
    const cleanName = companyName
      .replace(/\s+(Inc\.?|Corp\.?|Co\.?|Ltd\.?|LLC|Limited|Corporation)$/i, '')
      .replace(/[^\w\s]/g, '')
      .replace(/\s+/g, '')
      .toLowerCase();
    return `${cleanName}.com`;
  };

  const handleLogoError = (e) => {
    e.target.src = '/default-stock-logo.png';
  };

  const handleStockSelection = (selectedOption) => {
    const stockSymbol = selectedOption.value;
    if (!selectedStocks.includes(stockSymbol) && 
        selectedStocks.length < optimizationParams.shares) {
      setSelectedStocks(prev => [...prev, stockSymbol]);
    }
  };

  const removeStock = (stockToRemove) => {
    setSelectedStocks(prev => prev.filter(stock => stock !== stockToRemove));
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const result = await optimizePortfolio({
        mode: optimizationParams.mode,
        balance: parseFloat(optimizationParams.balance),
        shares: optimizationParams.shares,
        stocks: selectedStocks,
        ...optimizationParams.advancedParams
      });
      
      sessionStorage.setItem('optimizationResults', JSON.stringify(result));
      navigate('/optimize/results');
    } catch (err) {
      setError('Optimization failed');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Custom styles for react-select
  const customStyles = {
    control: (provided, state) => ({
      ...provided,
      backgroundColor: 'rgba(49, 53, 63, 1)',
      borderColor: state.isFocused ? 'rgba(57, 110, 247, 1)' : 'rgba(128, 128, 128, 1)',
      boxShadow: 'none',
      '&:hover': {
        borderColor: 'rgba(57, 110, 247, 1)'
      }
    }),
    menu: (provided) => ({
      ...provided,
      backgroundColor: 'rgba(27, 32, 40, 1)',
    }),
    option: (provided, state) => ({
      ...provided,
      backgroundColor: state.isFocused ? 'rgba(57, 110, 247, 0.1)' : 'transparent',
      color: 'white',
      display: 'flex',
      alignItems: 'center',
      padding: '8px 12px',
      cursor: 'pointer',
      '&:hover': {
        backgroundColor: 'rgba(57, 110, 247, 0.2)',
      }
    }),
    singleValue: (provided) => ({
      ...provided,
      color: 'white',
    }),
    input: (provided) => ({
      ...provided,
      color: 'white',
    }),
    placeholder: (provided) => ({
      ...provided,
      color: 'rgba(255, 255, 255, 0.5)',
    }),
  };

  // Format options for react-select
  const stockOptions = stocks
    .filter(stock => !selectedStocks.includes(stock.Symbol))
    .map(stock => ({
      value: stock.Symbol,
      label: (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <img
            src={stock.logoUrl}
            alt={`${stock.Security} logo`}
            style={{
              width: '24px',
              height: '24px',
              borderRadius: '50%',
              backgroundColor: 'white',
              padding: '2px'
            }}
            onError={handleLogoError}
          />
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span>{stock.Symbol}</span>
            <span style={{ fontSize: '0.8em', color: 'rgba(255, 255, 255, 0.7)' }}>
              {stock.Security}
            </span>
          </div>
        </div>
      )
    }));

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="main-content">
      <div className="header">
        <h2>Stock Selections</h2>
      </div>

      <div className="optimize-form">
        <div className="form-group">
          <label>Select Stocks ({selectedStocks.length}/{optimizationParams.shares}):</label>
          <Select
            options={stockOptions}
            styles={customStyles}
            onChange={handleStockSelection}
            isDisabled={selectedStocks.length >= optimizationParams.shares}
            placeholder="Search and select stocks..."
            className="stock-select"
            classNamePrefix="stock-select"
            isClearable
            isSearchable
            value={null}
          />
        </div>

        <div className="selected-stocks">
          {selectedStocks.map((stockSymbol) => {
            const stockInfo = stocks.find(s => s.Symbol === stockSymbol);
            return (
              <div key={stockSymbol} className="selected-stock-item">
                <div className="stock-info">
                  <img
                    src={stockInfo?.logoUrl}
                    alt={`${stockInfo?.Security} logo`}
                    className="stock-logo"
                    onError={handleLogoError}
                  />
                  <div className="stock-details">
                    <span className="stock-symbol">{stockSymbol}</span>
                    <span className="stock-name">{stockInfo?.Security}</span>
                  </div>
                </div>
                <button 
                  onClick={() => removeStock(stockSymbol)}
                  className="remove-stock-btn"
                  aria-label="Remove stock"
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>

        <div className="optimize-button">
          <button 
            onClick={handleSubmit}
            disabled={selectedStocks.length !== parseInt(optimizationParams.shares)}
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
};

export default OptimizeStocks;