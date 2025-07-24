# 📈 Portfolio Optimization using Financial Engineering Techniques

A modern, intelligent portfolio optimization platform that leverages machine learning to help investors build optimal stock portfolios based on S&P 500 companies.

> ✅ Includes: React.js frontend, Flask backend, REST API, scikit-learn ML models, and live financial data from Yahoo Finance.

---

## 🔍 Overview

This project aims to simplify intelligent investing by integrating financial engineering models with interactive web technologies.  
Users can create optimal stock portfolios based on expected return, risk tolerance, volatility constraints, and investment horizon either manually or with AI assistance.

---

## ✨ Key Features

- 🧠 **AI-Driven Portfolio Optimization**  
  Automatically selects stocks using RandomForest, Gradient Boosting, and K-Means clustering

- 📊 **Interactive Visualizations**  
  Animated pie charts, real-time stock graphs, and performance breakdowns

- ⚙️ **Manual & Auto Modes**  
  Choose stocks manually or let the system optimize based on AI models

- 🔐 **Advanced Constraints**  
  Set volatility range, allocation caps, investment horizon, and rebalance frequency

- 🌐 **Live Market Data**  
  Real-time S&P 500 price feeds, charts, and analytics via Yahoo Finance API

---

## 🛠️ Tech Stack

### Frontend
- React.js
- Chart.js / Visx
- React Router
- React Select
- Lottie React

### Backend
- Flask (Python)
- yfinance (financial data)
- scikit-learn (ML models)
- pandas, numpy, scipy

### Machine Learning
- Random Forest (return prediction)
- Gradient Boosting Classifier (stock labeling)
- K-Means Clustering (sector diversification)
- Technical Indicators: SMA, Bollinger Bands, ROC

---

## 🧠 Machine Learning Models

- **Feature Engineering**: P/E ratio, ROE, volatility, SMA, momentum, beta
- **Model Pipeline**:
  - Historical S&P 500 data collection
  - Feature extraction and normalization
  - Multi-model scoring & selection
  - Mean-variance optimization (with constraints)

---

## 🔧 REST API Endpoints (Flask Backend)

- `POST /api/optimize`  
  → Optimizes a portfolio with given constraints

- `GET /api/stocks`  
  → Returns stock list with metadata

- `GET /api/stock-data?symbol=AAPL&period=1y`  
  → Returns stock history for a given ticker

---

## 💻 How to Run

### 🔧 Prerequisites

- Node.js (v14+)
- Python (3.8+)
- pip

---

### ⚙️ Backend (Flask API)

```bash
cd backend/
pip install -r requirements.txt
python backend.py
# Runs at http://localhost:5000
```

### 🌐 Frontend (React)

```bash
cd frontend/
npm install
npm run dev
# Runs at http://localhost:3000
```

## 🎨 Demo & UI/UX

- 🎥 [Watch Demo Video](https://youtu.be/NJ9zv-oxqQM)  
- 🎨 [View Figma Prototype](https://www.figma.com/proto/igDh2CtU0KW0Sj3XMg41Oe/DS1?page-id=0%3A1&node-id=3-2074)  

---

## 📏 Performance Evaluation

- Expected Return (CAPM)
- Sharpe Ratio
- Volatility Analysis
- Allocation Breakdown
- Cross-validation & regression/classification metrics

---

## ⚠️ Disclaimer

This project is intended for **educational purposes only**.  
It is **not financial advice** and should **not be used** for live trading or real investment decisions.

---

## 📄 License

MIT License, Open source and free to use for learning or personal projects.

---

## 🙏 Acknowledgements

- Yahoo Finance API for market data  
- scikit-learn for ML capabilities  
- Flask and React for rapid development

