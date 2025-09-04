# Overview

QuantumTrade is an advanced cryptocurrency trading bot built in Python that combines technical analysis, machine learning, and risk management to automate trading decisions. The system features a web-based dashboard for monitoring and control, real-time alerts via email and Telegram, and comprehensive trade tracking. The bot now includes full Binance API integration for live trading with real market data, while maintaining fallback simulation capabilities for development and testing.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture
- **Core Framework**: Python-based modular architecture with asyncio for concurrent operations
- **Trading Engine**: Main `TradingBot` class orchestrates all trading operations and integrates multiple analysis engines
- **Data Management**: SQLite database with encryption for storing trades, configurations, and sensitive data
- **Configuration**: JSON-based configuration system with environment variable overrides for security

## Frontend Architecture
- **Web Interface**: Flask-based dashboard with Bootstrap 5 for responsive design
- **Real-time Updates**: JavaScript-based auto-refresh functionality for live data display
- **Visualization**: Chart.js integration for performance charts and trading analytics
- **Navigation**: Multi-page interface with dashboard and trade history views

## Data Processing Components
- **Technical Analysis**: Implements EMA, RSI, Moving Averages, and candlestick pattern recognition
- **Machine Learning**: Predictive model system (currently mocked for environments without ML libraries)
- **Risk Management**: Position sizing calculations using Kelly Criterion and volatility-based adjustments
- **Data Feed**: Binance API integration with automatic fallback to mock data generator
- **Live Trading**: Real cryptocurrency trade execution through Binance exchange
- **Market Data**: Real-time price feeds, order book data, and account balance management

## Security and Data Protection
- **Encryption**: Cryptography library with PBKDF2 key derivation for sensitive data protection
- **Environment Variables**: Secure handling of API keys, passwords, and tokens
- **Fallback Mechanisms**: Simple base64 encoding fallback when cryptography libraries unavailable

## Alert and Notification System
- **Multi-channel Alerts**: Email (SMTP) and Telegram bot integration
- **Trade Notifications**: Automated alerts for trade executions, profits, and losses
- **Configuration-driven**: Enable/disable specific alert channels via configuration

## Application Flow
- **Initialization**: Database setup, component initialization, and configuration loading
- **Trading Loop**: Continuous market analysis, signal generation, and trade execution
- **Web Interface**: Parallel Flask server for dashboard access and bot control
- **Data Persistence**: All trades and performance metrics stored in encrypted SQLite database

# External Dependencies

## Core Dependencies
- **Flask**: Web framework for dashboard interface
- **SQLite**: Embedded database for trade and configuration storage
- **Cryptography**: Data encryption and security (with fallback for environments without it)

## Optional ML Dependencies
- **NumPy/Pandas**: Data processing for machine learning features (mocked if unavailable)
- **Scikit-learn**: Machine learning algorithms (referenced but not strictly required)

## Alert Services
- **SMTP**: Email notifications via configurable SMTP servers (default: Gmail)
- **Telegram Bot API**: Real-time notifications via Telegram messaging

## Frontend Libraries (CDN-based)
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements
- **Chart.js**: JavaScript charting library for performance visualization

## Development and Testing
- **Mock Data Generation**: Built-in cryptocurrency market data simulation
- **Logging**: Python logging module with file and console output
- **Environment Configuration**: OS environment variables for sensitive configuration