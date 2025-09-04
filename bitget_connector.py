"""
Bitget Exchange Connector for QuantumTrade Bot
Handles real-time data feeds and trade execution via Bitget API
Enhanced with rate limiting, retry logic, and robust error handling
"""

import os
import time
import logging
import threading
import hashlib
import hmac
import base64
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps

try:
    import bitget
    from bitget.client import Client
    from bitget import consts as bc
    BITGET_AVAILABLE = True
except ImportError:
    BITGET_AVAILABLE = False
    Client = None
    bc = None
    logging.warning("Bitget library not available. Using mock data.")

from encryption import EncryptionManager

class RateLimiter:
    """Rate limiter for Bitget API calls"""

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

class BitgetConnector:
    """Real-time Bitget exchange connector for live trading"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.encryption = EncryptionManager()
        self.client = None
        self.rate_limiter = RateLimiter(max_calls_per_minute=1000)  # Bitget limit

        # Get bitget config with safe defaults
        bitget_config = {}
        if hasattr(config, 'get') and config is not None:
            bitget_config = config.get('bitget', {})
        elif isinstance(config, dict):
            bitget_config = config.get('bitget', {})

        self.is_testnet = bitget_config.get('testnet', True)
        self.min_notional = bitget_config.get('min_order_value', 5.0)
        self.max_retries = bitget_config.get('max_retries', 3)
        self.retry_delay = bitget_config.get('retry_delay', 1.0)

        # Base URLs
        if self.is_testnet:
            self.base_url = "https://api.bitget.com"
        else:
            self.base_url = "https://api.bitget.com"

        # Initialize Bitget client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Bitget API client with credentials"""
        try:
            # Get API credentials from environment
            api_key = os.getenv('BITGET_API_KEY')
            secret_key = os.getenv('BITGET_SECRET_KEY')
            passphrase = os.getenv('BITGET_PASSPHRASE')

            if not api_key or not secret_key or not passphrase:
                self.logger.warning("Bitget API credentials not found. Using mock data.")
                return

            if not BITGET_AVAILABLE:
                self.logger.warning("Bitget library not installed. Using mock data.")
                return

            # Initialize client
            self.client = Client(api_key, secret_key, passphrase)

            # Test connection
            self._test_connection()

        except Exception as e:
            self.logger.error(f"Failed to initialize Bitget client: {e}")
            self.client = None

    def _test_connection(self):
        """Test API connection and permissions"""
        try:
            if not self.client:
                return False

            # Test connectivity by getting server time
            response = self.client._get('/api/spot/v1/public/time')
            if response.get('code') == '00000':
                self.logger.info("Bitget API connection successful")
                return True
            else:
                self.logger.error(f"Bitget API test failed: {response}")
                return False

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate Bitget API signature"""
        if not self.client:
            return ""

        secret_key = os.getenv('BITGET_SECRET_KEY', '')
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).digest()
        return base64.b64encode(signature).decode('utf-8')

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def get_market_data(self, symbol: str = 'BTCUSDT') -> Dict:
        """Get real-time market data from Bitget with rate limiting and retry"""
        try:
            if not self.client:
                return self._get_mock_data(symbol)

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Get ticker data
            ticker_response = self.client.spot_get_ticker(symbol)
            if ticker_response.get('code') != '00000':
                raise Exception(f"Failed to get ticker: {ticker_response}")

            ticker_data = ticker_response['data']

            # Get recent klines for technical analysis
            klines_response = self.client.spot_get_candles(symbol, '1m', 100)
            klines = []
            if klines_response.get('code') == '00000':
                klines = klines_response['data']

            # Format market data
            current_price = float(ticker_data['last'])

            return {
                'symbol': symbol,
                'price': current_price,
                'bid': float(ticker_data.get('bid', current_price * 0.999)),
                'ask': float(ticker_data.get('ask', current_price * 1.001)),
                'volume': float(ticker_data.get('volume', 0)),
                'high_24h': float(ticker_data.get('high24h', current_price * 1.02)),
                'low_24h': float(ticker_data.get('low24h', current_price * 0.98)),
                'change_24h': float(ticker_data.get('change', 0)),
                'timestamp': int(time.time() * 1000),
                'klines': self._format_klines(klines),
                'source': 'bitget_live'
            }

        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return self._get_mock_data(symbol)

    def _format_klines(self, klines: List) -> List[Dict]:
        """Format kline data for technical analysis"""
        formatted = []
        for kline in klines:
            if isinstance(kline, list) and len(kline) >= 6:
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
        """Execute real trade on Bitget with rate limiting and retry"""
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
            order_type = 'market' if price is None else 'limit'
            side = 'buy' if action.lower() == 'buy' else 'sell'

            order_params = {
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'size': str(quantity),
            }

            if price:
                order_params['price'] = str(price)

            # Execute order
            if order_type == 'market':
                response = self.client.spot_post_order(order_params)
            else:
                response = self.client.spot_post_order(order_params)

            if response.get('code') == '00000':
                order_data = response['data']
                self.logger.info(f"Trade executed: {action} {quantity} {symbol} - Order ID: {order_data.get('orderId')}")

                return {
                    'success': True,
                    'order_id': order_data.get('orderId', 'N/A'),
                    'symbol': symbol,
                    'side': action,
                    'quantity': quantity,
                    'price': float(order_data.get('price', current_price)),
                    'status': order_data.get('status', 'filled'),
                    'timestamp': int(time.time() * 1000),
                    'commission': self._calculate_commission(order_value),
                    'source': 'bitget_live'
                }
            else:
                raise Exception(f"Order failed: {response}")

        except Exception as e:
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

            response = self.client.spot_get_account_assets()
            if response.get('code') == '00000':
                for balance in response['data']:
                    if balance.get('coinName') == asset:
                        return float(balance.get('available', 0))
            return 0.0

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0

    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        try:
            if not self.client:
                return []

            response = self.client.spot_get_open_orders(symbol=symbol)
            if response.get('code') == '00000':
                return response['data']
            return []

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            if not self.client:
                return False

            response = self.client.spot_cancel_order(symbol, order_id)
            if response.get('code') == '00000':
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return False

    def _get_mock_data(self, symbol: str) -> Dict:
        """Fallback mock data when Bitget is unavailable"""
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

            # Bitget doesn't have detailed symbol info in their public API
            # Return default values
            return {
                'min_qty': 0.0001,
                'max_qty': 10000,
                'step_size': 0.0001,
                'min_notional': self.min_notional
            }

        except Exception as e:
            self.logger.error(f"Failed to get symbol info: {e}")
            return {}

    def is_connected(self) -> bool:
        """Check if connected to Bitget API"""
        return self.client is not None and BITGET_AVAILABLE

    def get_connection_status(self) -> Dict:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected(),
            'testnet': self.is_testnet,
            'library_available': BITGET_AVAILABLE,
            'credentials_provided': bool(
                os.getenv('BITGET_API_KEY') and
                os.getenv('BITGET_SECRET_KEY') and
                os.getenv('BITGET_PASSPHRASE')
            )
        }
