
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export const fetchStocks = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/stocks`);
    if (!response.ok) throw new Error('Failed to fetch stocks');
    return await response.json();
  } catch (error) {
    console.error('Error fetching stocks:', error);
    throw error;
  }
};

export const optimizePortfolio = async (optimizationParams) => {
  try {

    console.log('Making API request with params:', optimizationParams);

    const response = await fetch(`${API_BASE_URL}/optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(optimizationParams),
    });

    console.log('API response status:', response.status);

    if (!response.ok) {
      const errorData = await response.text();
      console.error('API error response:', errorData); // Debug log
      throw new Error(`Optimization failed: ${errorData}`);
    }

      const data = await response.json();
      console.log('API response data:', data);
      // Return the allocations array directly
      return data.allocations || data;
    } catch (error) {
      console.error('API call error:', error);
      throw error;
    }
  };

export const fetchStockData = async (symbol, period = '1d') => {
  try {
    const response = await fetch(`${API_BASE_URL}/stock-data?symbol=${symbol}&period=${period}`);
    if (!response.ok) throw new Error('Failed to fetch stock data');
    return await response.json();
  } catch (error) {
    console.error('Error fetching stock data:', error);
    throw error;
  }
};

export const fetchSP500Details = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/sp500-details`);
    if (!response.ok) throw new Error('Failed to fetch S&P 500 details');
    return await response.json();
  } catch (error) {
    console.error('Error fetching S&P 500 details:', error);
    throw error;
  }
};
