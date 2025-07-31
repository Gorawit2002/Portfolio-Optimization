// src/utils/logoUtils.js
const specialLogoMap = {
    'GOOGL': 'google',
    'GOOG': 'google',
    'ABBV': 'abbvie',
    'ADBE': 'adobe',
    'MMM': '3m',
    'AMZN': 'amazon',
    'AAPL': 'apple',
    'MSFT': 'microsoft',
    'META': 'meta',
    'BRK.B': 'berkshirehathaway',
    'BRK-B': 'berkshirehathaway',
    // Add more special cases as needed
  };
  
  export const getLogoUrl = (symbol, companyName) => {
    // Check for special cases first
    if (specialLogoMap[symbol]) {
      return `https://logo.clearbit.com/${specialLogoMap[symbol]}.com`;
    }
  
    // Clean company name
    const cleanCompanyName = companyName
      .toLowerCase()
      .replace(/\s+(inc\.?|corp\.?|co\.?|ltd\.?|llc|limited|corporation|company)$/i, '')
      .replace(/[^a-z0-9]/g, '')
      .replace(/^the/, '');
  
    return `https://logo.clearbit.com/${cleanCompanyName}.com`;
  };
  
  export const StockLogoContainer = ({ symbol, companyName, children }) => {
    return (
      <div className="stock-logo-container" style={{ backgroundColor: '#000000', borderRadius: '20px', padding: '4px 10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
        {children}
      </div>
    );
  };