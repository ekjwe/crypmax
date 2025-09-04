"""
Binance Exchange Connector for QuantumTrade Bot
Handles real-time data feeds and trade execution via Binance API
Enhanced with rate limiting, retry logic, and robust error handling
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import requests
from functools import wraps

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    from binance.enums import (
        SIDE_BUY, SIDE_SELL,
        ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT,
        TIME_IN_FORCE_GTC
    )
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception
    BinanceRequestException = Exception
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    TIME_IN_FORCE_GTC = "GTC"
    logging.warning("Binance library not available. Using mock data.")

from encryption import EncryptionManager

class RateLimiter:
    """Rate limiter for Binance API calls"""

    def __init__(self, max_calls_per_minute: int = 1200):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [call for call in self.calls if now - call < 60]

            if len(self.calls) >= self.max_calls_per_minute:
                # Wait until oldest call expires
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    self.calls = [call for call in self.calls if time.time() - call < 60]

            self.calls.append(now)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying API calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (BinanceAPIException, BinanceRequestException, requests.exceptions.RequestException) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"API call failed after {max_retries + 1} attempts: {e}")
                        raise e
                except Exception as e:
                    # Don't retry for non-API related errors
                    logging.error(f"Non-retryable error: {e}")
                    raise e

            raise last_exception
        return wrapper
    return decorator

class BinanceConnector:
    """Real-time Binance exchange connector for live trading"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.encryption = EncryptionManager()
        self.client = None
        self.rate_limiter = RateLimiter(max_calls_per_minute=1200)  # Binance limit

        # Get binance config with safe defaults
        binance_config = {}
        if hasattr(config, 'get') and config is not None:
            binance_config = config.get('binance', {})
        elif isinstance(config, dict):
            binance_config = config.get('binance', {})

        self.is_testnet = binance_config.get('testnet', True)
        self.min_notional = binance_config.get('min_order_value', 10.0)
        self.max_retries = binance_config.get('max_retries', 3)
        self.retry_delay = binance_config.get('retry_delay', 1.0)

        # Initialize Binance client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize Binance API client with credentials"""
        try:
            # Get API credentials from environment
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if not api_key or not secret_key:
                self.logger.warning("Binance API credentials not found. Using mock data.")
                return
                
            if not BINANCE_AVAILABLE:
                self.logger.warning("Binance library not installed. Using mock data.")
                return
                
            # Initialize client
            self.client = Client(
                api_key=api_key,
                api_secret=secret_key,
                testnet=self.is_testnet
            )
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance client: {e}")
            self.client = None
            
    def _test_connection(self):
        """Test API connection and permissions"""
        try:
            if not self.client:
                return False
                
            # Test connectivity
            status = self.client.get_system_status()
            self.logger.info(f"Binance system status: {status}")
            
            # Test account access
            account = self.client.get_account()
            self.logger.info(f"Account status: {account.get('accountType', 'Unknown')}")
            
            # Log available balances
            balances = [b for b in account['balances'] if float(b['free']) > 0]
            self.logger.info(f"Available balances: {len(balances)} assets")
            
            return True
            
        except Exception as e:
            if BINANCE_AVAILABLE and 'BinanceAPIException' in str(type(e)):
                self.logger.error(f"Binance API error: {e}")
            else:
                self.logger.error(f"Connection test failed: {e}")
            return False
            
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def get_market_data(self, symbol: str = 'BTCUSDT') -> Dict:
        """Get real-time market data from Binance with rate limiting and retry"""
        try:
            if not self.client:
                return self._get_mock_data(symbol)

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Get 24hr ticker statistics
            ticker = self.client.get_ticker(symbol=symbol)

            # Get recent klines for technical analysis
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=100
            ) if hasattr(Client, 'KLINE_INTERVAL_1MINUTE') else []

            # Format market data
            current_price = float(ticker['lastPrice'])

            return {
                'symbol': symbol,
                'price': current_price,
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'volume': float(ticker['volume']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'change_24h': float(ticker['priceChangePercent']),
                'timestamp': int(time.time() * 1000),
                'klines': self._format_klines(klines),
                'source': 'binance_live'
            }

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return self._get_mock_data(symbol)
            
    def _format_klines(self, klines: List) -> List[Dict]:
        """Format kline data for technical analysis"""
        formatted = []
        for kline in klines:
            formatted.append({
                'timestamp': int(kline[0]),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        return formatted
        
    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def execute_trade(self, action: str, symbol: str, quantity: float, price: float = None) -> Dict:
        """Execute real trade on Binance with rate limiting and retry"""
        try:
            if not self.client:
                return self._simulate_trade(action, symbol, quantity, price)

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Validate minimum order value
            if price is not None:
                current_price = price
            else:
                ticker_data = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker_data['price'])
            order_value = quantity * current_price

            if order_value < self.min_notional:
                raise ValueError(f"Order value ${order_value:.2f} below minimum ${self.min_notional}")

            # Prepare order parameters
            side = SIDE_BUY if action.lower() == 'buy' else SIDE_SELL
            order_type = ORDER_TYPE_MARKET if price is None else ORDER_TYPE_LIMIT

            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': f"{quantity:.6f}",
            }

            if price:
                order_params['price'] = f"{price:.2f}"
                order_params['timeInForce'] = TIME_IN_FORCE_GTC

            # Execute order
            order = self.client.create_order(**order_params)

            self.logger.info(f"Trade executed: {action} {quantity} {symbol} - Order ID: {order['orderId']}")

            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': action,
                'quantity': quantity,
                'price': float(order.get('price', current_price)),
                'status': order['status'],
                'timestamp': order['transactTime'],
                'commission': self._calculate_commission(order_value),
                'source': 'binance_live'
            }

        except Exception as e:
            if BINANCE_AVAILABLE and 'BinanceAPIException' in str(type(e)):
                self.logger.error(f"Binance API error during trade: {e}")
            else:
                self.logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
            
    def _calculate_commission(self, order_value: float) -> float:
        """Calculate trading commission (0.1% for regular users)"""
        return order_value * 0.001
        
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get account balance for specific asset"""
        try:
            if not self.client:
                return 1000.0  # Mock balance
                
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0
            
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            if not self.client:
                return []
                
            orders = self.client.get_open_orders(symbol=symbol if symbol else None)
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
            
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel an open order"""
        try:
            if not self.client:
                return False
                
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False
            
    def _get_mock_data(self, symbol: str) -> Dict:
        """Fallback mock data when Binance is unavailable"""
        import random
        base_price = 45000 if 'BTC' in symbol else 2500
        current_price = base_price + random.uniform(-1000, 1000)
        
        return {
            'symbol': symbol,
            'price': current_price,
            'bid': current_price - 0.01,
            'ask': current_price + 0.01,
            'volume': random.uniform(1000, 5000),
            'high_24h': current_price + random.uniform(100, 500),
            'low_24h': current_price - random.uniform(100, 500),
            'change_24h': random.uniform(-5, 5),
            'timestamp': int(time.time() * 1000),
            'klines': [],
            'source': 'mock_fallback'
        }
        
    def _simulate_trade(self, action: str, symbol: str, quantity: float, price: float) -> Dict:
        """Simulate trade execution for testing"""
        return {
            'success': True,
            'order_id': f"MOCK_{int(time.time())}",
            'symbol': symbol,
            'side': action,
            'quantity': quantity,
            'price': price or 45000,
            'status': 'FILLED',
            'timestamp': int(time.time() * 1000),
            'commission': 0.1,
            'source': 'mock_simulation'
        }
        
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading rules and filters for a symbol"""
        try:
            if not self.client:
                return {
                    'min_qty': 0.001,
                    'max_qty': 1000,
                    'step_size': 0.001,
                    'min_notional': self.min_notional
                }
                
            exchange_info = self.client.get_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    filters = {f['filterType']: f for f in symbol_info['filters']}
                    
                    return {
                        'min_qty': float(filters.get('LOT_SIZE', {}).get('minQty', 0.001)),
                        'max_qty': float(filters.get('LOT_SIZE', {}).get('maxQty', 1000)),
                        'step_size': float(filters.get('LOT_SIZE', {}).get('stepSize', 0.001)),
                        'min_notional': float(filters.get('MIN_NOTIONAL', {}).get('minNotional', self.min_notional))
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get symbol info: {e}")
            return {}
            
    def is_connected(self) -> bool:
        """Check if connected to Binance API"""
        return self.client is not None and BINANCE_AVAILABLE
        
    def get_connection_status(self) -> Dict:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected(),
            'testnet': self.is_testnet,
            'library_available': BINANCE_AVAILABLE,
            'credentials_provided': bool(os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_SECRET_KEY'))
        }