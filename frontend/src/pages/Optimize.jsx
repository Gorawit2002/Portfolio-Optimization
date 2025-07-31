import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { optimizePortfolio } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

const Optimize = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    mode: 'auto',
    balance: '',
    shares: 2,
    showAdvanced: false,
    advancedParams: {
      investmentHorizon: 'short',
      targetVolatility: '',
      maxVolatility: '',
      minVolatility: '',
      maxAllocation: '',
      minAllocation: '',
      rebalanceFrequency: 'monthly'
    }
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (type === 'checkbox') {
      setFormData(prev => ({ ...prev, [name]: checked }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleAdvancedParamChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      advancedParams: {
        ...prev.advancedParams,
        [name]: value
      }
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Starting optimization process...'); // Debug log
    console.log('Form data:', formData); // Debug log
        // Check if the balance is negative
    if (parseFloat(formData.balance) < 0) {
      alert('The balance cannot be negative.');
      return; // Stop the form submission
    }
    
    if (formData.mode === 'auto') {
      try {
        setLoading(true);
        // Direct API call for auto mode
        console.log('Making API call with params:', {  // Debug log
          mode: 'auto',
          balance: parseFloat(formData.balance),
          shares: formData.shares,
          ...formData.advancedParams
        });

        const result = await optimizePortfolio({
          mode: 'auto',
          balance: parseFloat(formData.balance),
          shares: formData.shares,
          ...formData.advancedParams
        });
        console.log('API response:', result);

        // Store results and navigate to results page
        sessionStorage.setItem('optimizationResults', JSON.stringify(result));
        console.log('Stored in sessionStorage:', JSON.stringify(result)); 
        navigate('/optimize/results');
      } catch (err) {
        console.error('Optimization error:', err);
        setError('Optimization failed');
        console.error(err);
      } finally {
        setLoading(false);
      }
    } else {
      // For manual mode, store params and go to stock selection page
      sessionStorage.setItem('optimizationParams', JSON.stringify(formData));
      navigate('/optimize/stocks');
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="main-content">
      <div className="header">
        <h1>Optimize</h1>
      </div>
      
      <form className="optimize-form" onSubmit={handleSubmit}>
        <h2>Optimizer Settings</h2>
        
        <div className="form-group">
          <label>Index</label>
          <select disabled>
            <option>S&P 500</option>
          </select>
        </div>

        <div className="form-group">
          <label>Number of Shares</label>
          <input
            type="number"
            name="shares"
            value={formData.shares}
            onChange={handleChange}
            min="2"
            max="10"
          />
        </div>

        <div className="balance-display">
          <label>Your Balance ($)</label>
          <input
            type="number"
            name="balance"
            value={formData.balance}
            onChange={handleChange}
            placeholder="Enter your investment amount"
            required
          />
        </div>

        <div className="mode-options">
          <label>
            <input
              type="radio"
              name="mode"
              value="auto"
              checked={formData.mode === 'auto'}
              onChange={handleChange}
            />
            Auto
          </label>
          <label>
            <input
              type="radio"
              name="mode"
              value="manual"
              checked={formData.mode === 'manual'}
              onChange={handleChange}
            />
            Manual
          </label>
        </div>

        <div className="advanced-parameter">
          <label>
            <input
              type="checkbox"
              name="showAdvanced"
              checked={formData.showAdvanced}
              onChange={handleChange}
            />
            Advanced Parameter
          </label>
        </div>

        {formData.showAdvanced && (
          <div id="advancedInputs">
            <div className="investment-horizon">
              <label>Investment Horizon:</label>
              <div className="horizon-options">
                {[
                  { value: 'short', label: 'Short (1-3 years)' },
                  { value: 'medium', label: 'Medium (4-6 years)' },
                  { value: 'long', label: 'Long (10+ years)' }
                ].map(({ value, label }) => (
                  <label key={value}>
                    <input
                      type="radio"
                      name="investmentHorizon"
                      value={value}
                      checked={formData.advancedParams.investmentHorizon === value}
                      onChange={handleAdvancedParamChange}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>

            <div className="volatility-range">
              <div className="form-group">
                <label>Max Volatility (%):</label>
                <input
                  type="number"
                  name="maxVolatility"
                  value={formData.advancedParams.maxVolatility}
                  onChange={handleAdvancedParamChange}
                />
              </div>
              <div className="form-group">
                <label>Min Volatility (%):</label>
                <input
                  type="number"
                  name="minVolatility"
                  value={formData.advancedParams.minVolatility}
                  onChange={handleAdvancedParamChange}
                />
              </div>
            </div>

            <div className="allocation-range">
              <div className="form-group">
                <label>Max Allocation (%):</label>
                <input
                  type="number"
                  name="maxAllocation"
                  value={formData.advancedParams.maxAllocation}
                  onChange={handleAdvancedParamChange}
                />
              </div>
              <div className="form-group">
                <label>Min Allocation (%):</label>
                <input
                  type="number"
                  name="minAllocation"
                  value={formData.advancedParams.minAllocation}
                  onChange={handleAdvancedParamChange}
                />
              </div>
            </div>

            <div className="form-group">
              <label>Rebalance:</label>
              <div className="rebalance-options">
                {['monthly', 'quarterly', 'annually'].map(option => (
                  <label key={option}>
                    <input
                      type="radio"
                      name="rebalanceFrequency"
                      value={option}
                      checked={formData.advancedParams.rebalanceFrequency === option}
                      onChange={handleAdvancedParamChange}
                    />
                    {option.charAt(0).toUpperCase() + option.slice(1)}
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        <div className="optimize-button">
          <button type="submit">
            {formData.mode === 'auto' ? 'Optimize' : 'Next'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Optimize;