"""
Coinbase Advanced API Exchange Connector for QuantumTrade Bot
Handles real-time data feeds and trade execution via Coinbase Advanced API
Enhanced with rate limiting, retry logic, and robust error handling
"""

import os
import time
import logging
import threading
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps

try:
    import requests
except ImportError:
    requests = None

try:
    from coinbase_advanced_py import CoinbaseAdvancedTradeAPI, init
    COINBASE_AVAILABLE = True
except ImportError:
    COINBASE_AVAILABLE = False
    CoinbaseAdvancedTradeAPI = None
    init = None
    logging.warning("Coinbase Advanced API library not available. Using mock data.")

from encryption import EncryptionManager

class RateLimiter:
    """Rate limiter for Coinbase API calls"""

    def __init__(self, max_calls_per_minute: int = 1000):
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
                except (requests.exceptions.RequestException, Exception) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(f"API call failed after {max_retries + 1} attempts: {e}")
                        raise e

            raise last_exception
        return wrapper
    return decorator

class CoinbaseConnector:
    """Real-time Coinbase Advanced API exchange connector for live trading"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.encryption = EncryptionManager()
        self.client = None
        self.rate_limiter = RateLimiter(max_calls_per_minute=1000)  # Coinbase limit

        # Get coinbase config with safe defaults
        coinbase_config = {}
        if hasattr(config, 'get') and config is not None:
            coinbase_config = config.get('coinbase', {})
        elif isinstance(config, dict):
            coinbase_config = config.get('coinbase', {})

        self.is_sandbox = coinbase_config.get('sandbox', True)
        self.min_notional = coinbase_config.get('min_order_value', 5.0)
        self.max_retries = coinbase_config.get('max_retries', 3)
        self.retry_delay = coinbase_config.get('retry_delay', 1.0)

        # Initialize Coinbase client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Coinbase Advanced API client with credentials"""
        try:
            api_key = os.getenv('COINBASE_API_KEY')
            api_secret = os.getenv('COINBASE_SECRET_KEY')

            if not api_key or not api_secret:
                self.logger.warning("Coinbase API credentials not found. Using mock data.")
                return

            if not COINBASE_AVAILABLE:
                self.logger.warning("Coinbase Advanced API library not installed. Using mock data.")
                return

            # Initialize client
            self.client = CoinbaseAdvancedTradeAPI(api_key, api_secret, is_sandbox=self.is_sandbox)

            # Test connection
            self._test_connection()

        except Exception as e:
            self.logger.error(f"Failed to initialize Coinbase client: {e}")
            self.client = None

    def _test_connection(self):
        """Test API connection and permissions"""
        try:
            if not self.client:
                return False

            # Test connectivity by getting accounts
            accounts = self.client.get_accounts()
            if accounts and hasattr(accounts, 'accounts'):
                self.logger.info("Coinbase Advanced API connection successful")
                return True
            else:
                self.logger.error(f"Coinbase API test failed: {accounts}")
                return False

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def get_market_data(self, symbol: str = 'BTC-USD') -> Dict:
        """Get real-time market data from Coinbase with rate limiting and retry"""
        try:
            if not self.client:
                return self._get_mock_data(symbol)

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Get ticker data
            ticker = self.client.get_product_ticker(symbol)
            if not ticker or 'price' not in ticker:
                raise Exception(f"Failed to get ticker: {ticker}")

            # Coinbase Advanced API may provide klines, but we'll mock for now
            klines = []

            current_price = float(ticker['price'])

            return {
                'symbol': symbol,
                'price': current_price,
                'bid': float(ticker.get('bid', current_price * 0.999)),
                'ask': float(ticker.get('ask', current_price * 1.001)),
                'volume': float(ticker.get('volume', 0)),
                'high_24h': float(ticker.get('high_24h', current_price * 1.02)),
                'low_24h': float(ticker.get('low_24h', current_price * 0.98)),
                'change_24h': float(ticker.get('change_24h', 0)),
                'timestamp': int(time.time() * 1000),
                'klines': klines,
                'source': 'coinbase_live'
            }

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return self._get_mock_data(symbol)

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def execute_trade(self, action: str, symbol: str, quantity: float, price: float = None) -> Dict:
        """Execute real trade on Coinbase with rate limiting and retry"""
        try:
            if not self.client:
                return self._simulate_trade(action, symbol, quantity, price)

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Get current price if not provided
            if price is None:
                market_data = self.get_market_data(symbol)
                current_price = market_data['price']
            else:
                current_price = price

            # Validate minimum order value
            order_value = quantity * current_price
            if order_value < self.min_notional:
                raise ValueError(f"Order value ${order_value:.2f} below minimum ${self.min_notional}")

            # Prepare order parameters
            side = 'BUY' if action.lower() == 'buy' else 'SELL'
            order_type = 'MARKET' if price is None else 'LIMIT'

            order_params = {
                'product_id': symbol,
                'side': side,
                'order_configuration': {
                    'market_market_ioc': {
                        'quote_size': str(order_value)
                    } if order_type == 'MARKET' else {
                        'limit_limit_gtc': {
                            'base_size': str(quantity),
                            'limit_price': str(price)
                        }
                    }
                }
            }

            # Execute order
            response = self.client.create_order(**order_params)

            if response and response.get('success'):
                order_data = response.get('order', {})
                self.logger.info(f"Trade executed: {action} {quantity} {symbol} - Order ID: {order_data.get('order_id')}")

                return {
                    'success': True,
                    'order_id': order_data.get('order_id', 'N/A'),
                    'symbol': symbol,
                    'side': action,
                    'quantity': quantity,
                    'price': float(order_data.get('average_filled_price', current_price)),
                    'status': order_data.get('status', 'filled'),
                    'timestamp': int(time.time() * 1000),
                    'commission': self._calculate_commission(order_value),
                    'source': 'coinbase_live'
                }
            else:
                raise Exception(f"Order failed: {response}")

        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_commission(self, order_value: float) -> float:
        """Calculate trading commission (0.6% for Coinbase Advanced)"""
        return order_value * 0.006

    def get_account_balance(self, asset: str = 'USD') -> float:
        """Get account balance for specific asset"""
        try:
            if not self.client:
                return 1000.0  # Mock balance

            accounts = self.client.get_accounts()
            if accounts and hasattr(accounts, 'accounts'):
                for account in accounts.accounts:
                    if account.currency == asset:
                        return float(account.available_balance.get('value', 0))
            return 0.0

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            if not self.client:
                return []

            orders = self.client.get_orders(product_id=symbol)
            if orders and hasattr(orders, 'orders'):
                return [order.__dict__ for order in orders.orders]
            return []

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            if not self.client:
                return False

            response = self.client.cancel_order(order_id)
            if response and response.get('success'):
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False

    def _get_mock_data(self, symbol: str) -> Dict:
        """Fallback mock data when Coinbase is unavailable"""
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

    def is_connected(self) -> bool:
        """Check if connected to Coinbase API"""
        return self.client is not None and COINBASE_AVAILABLE

    def get_connection_status(self) -> Dict:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected(),
            'sandbox': self.is_sandbox,
            'library_available': COINBASE_AVAILABLE,
            'credentials_provided': bool(
                os.getenv('COINBASE_API_KEY') and
                os.getenv('COINBASE_SECRET_KEY')
            )
        }
