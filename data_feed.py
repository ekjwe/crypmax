"""
Data Feed Manager for QuantumTrade Bot
Handles multiple exchange APIs with failover and load balancing
"""

import random
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import logging
from binance_connector import BinanceConnector
from bitget_connector import BitgetConnector
from coinbase_connector import CoinbaseConnector

class DataFeed:
    """Manages data feeds from Binance API with fallback to mock data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize exchange connectors
        self.exchanges = {}
        self._initialize_exchanges()

        # Current active exchange
        self.current_exchange = self._get_default_exchange()

        # Mock data properties for fallback
        self.current_prices = {
            "BTC/USD": 45000.0,
            "ETH/USD": 3200.0,
            "BNB/USD": 320.0,
            "DOGE/USD": 0.12
        }
        self.base_price = 45000  # Base BTC price
        self.current_price = self.base_price
        self.volatility_config = {
            "BTC/USD": 0.02,
            "ETH/USD": 0.025,
            "BNB/USD": 0.03,
            "DOGE/USD": 0.05
        }
        self.trend = 0.0001  # Slight upward trend
        self.running = False

        # Price history for technical analysis
        self.price_history = {}
        self.volume_history = []

        # Symbol mapping for different exchanges
        self.symbol_maps = {
            'binance': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'BNB/USD': 'BNBUSDT',
                'DOGE/USD': 'DOGEUSDT'
            },
            'bitget': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'BNB/USD': 'BNBUSDT',
                'DOGE/USD': 'DOGEUSDT'
            },
            'coinbase': {
                'BTC/USD': 'BTC-USD',
                'ETH/USD': 'ETH-USD',
                'BNB/USD': 'BNB-USD',
                'DOGE/USD': 'DOGE-USD'
            }
        }

        # Initialize price history
        self._initialize_price_history()

        # Check connection status
        self._log_connection_status()

    def _initialize_exchanges(self):
        """Initialize all enabled exchange connectors"""
        try:
            # Initialize Binance
            if self.config.get('exchanges', {}).get('binance', {}).get('enabled', True):
                self.exchanges['binance'] = BinanceConnector(self.config)
                self.logger.info("Binance connector initialized")

            # Initialize Bitget
            if self.config.get('exchanges', {}).get('bitget', {}).get('enabled', False):
                self.exchanges['bitget'] = BitgetConnector(self.config)
                self.logger.info("Bitget connector initialized")

            # Initialize Coinbase
            if self.config.get('exchanges', {}).get('coinbase', {}).get('enabled', False):
                self.exchanges['coinbase'] = CoinbaseConnector(self.config)
                self.logger.info("Coinbase connector initialized")

        except Exception as e:
            self.logger.error(f"Error initializing exchanges: {e}")

    def _get_default_exchange(self) -> str:
        """Get the default exchange from config"""
        return self.config.get('trading', {}).get('default_exchange', 'binance')

    def set_current_exchange(self, exchange: str) -> bool:
        """Set the active exchange"""
        if exchange in self.exchanges and self.exchanges[exchange].is_connected():
            self.current_exchange = exchange
            self.logger.info(f"Switched to exchange: {exchange}")
            return True
        else:
            self.logger.warning(f"Exchange {exchange} not available or not connected")
            return False

    def get_available_exchanges(self) -> List[str]:
        """Get list of available and connected exchanges"""
        return [exchange for exchange, connector in self.exchanges.items() if connector.is_connected()]

    def _get_exchange_connector(self, exchange: str = None):
        """Get connector for specified exchange or current exchange"""
        exchange = exchange or self.current_exchange
        return self.exchanges.get(exchange)

    def _log_connection_status(self):
        """Log current connection status for all exchanges"""
        for exchange_name, connector in self.exchanges.items():
            status = connector.get_connection_status()
            if status['connected']:
                mode = "TESTNET" if status.get('testnet', status.get('sandbox', False)) else "LIVE"
                self.logger.info(f"✓ Connected to {exchange_name.upper()} API ({mode})")
            else:
                self.logger.warning(f"⚠ {exchange_name.upper()} API not available")
                if not status['library_available']:
                    self.logger.warning(f"  - {exchange_name.upper()} library not installed")
                if not status['credentials_provided']:
                    self.logger.warning(f"  - {exchange_name.upper()} API credentials not provided")

        if not self.exchanges:
            self.logger.warning("⚠ No exchanges configured - using mock data only")
        
    def _initialize_price_history(self):
        """Initialize historical price data for all pairs"""
        for pair in self.current_prices:
            self.price_history[pair] = {}
            for timeframe in ["1m", "3m", "15m", "30m", "1h", "4h", "1d"]:
                self.price_history[pair][timeframe] = self._generate_historical_data(
                    pair, timeframe, 200
                )
    
    def _generate_historical_data(self, pair: str, timeframe: str, count: int) -> List[Dict]:
        """Generate historical OHLCV data"""
        data = []
        base_price = self.current_prices[pair]
        volatility = self.volatility_config[pair]
        
        # Convert timeframe to minutes
        timeframe_minutes = self._timeframe_to_minutes(timeframe)
        
        # Generate data points going backward in time
        for i in range(count, 0, -1):
            timestamp = datetime.now() - timedelta(minutes=i * timeframe_minutes)
            
            # Generate realistic OHLCV data
            open_price = base_price * (1 + random.gauss(0, volatility))
            
            # Generate high/low based on volatility
            high_mult = 1 + abs(random.gauss(0, volatility * 0.5))
            low_mult = 1 - abs(random.gauss(0, volatility * 0.5))
            
            high_price = open_price * high_mult
            low_price = open_price * low_mult
            
            # Close price within high/low range
            close_range = high_price - low_price
            close_price = low_price + (random.random() * close_range)
            
            # Volume simulation
            volume = random.uniform(100, 1000)
            
            data.append({
                "timestamp": int(timestamp.timestamp() * 1000),
                "datetime": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            })
            
        return sorted(data, key=lambda x: x["timestamp"])
        
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080
        }
        return mapping.get(timeframe, 60)
        
    def get_current_data(self, symbol: str = 'BTC/USD') -> Dict:
        """Get current market data from current exchange - real or mock"""

        # Get current exchange connector
        connector = self._get_exchange_connector()
        if connector and connector.is_connected():
            try:
                # Convert symbol to exchange format
                exchange_symbol = self.symbol_maps.get(self.current_exchange, {}).get(symbol, symbol)

                market_data = connector.get_market_data(exchange_symbol)

                # Convert back to standard format
                converted_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'price': market_data['price'],
                    'bid': market_data.get('bid', None),
                    'ask': market_data.get('ask', None),
                    'volume': market_data.get('volume', None),
                    'change_24h': market_data.get('change_24h', 0),
                    'high_24h': market_data.get('high_24h', None),
                    'low_24h': market_data.get('low_24h', None),
                    'market_cap': market_data['price'] * 19000000,  # Approximate BTC supply
                    'volatility': abs(market_data.get('change_24h', 0)) / 100,
                    'source': f'{self.current_exchange}_live',
                    'exchange': self.current_exchange,
                    'klines': market_data.get('klines', [])
                }

                # Update price history with real data
                self._update_price_history(converted_data, symbol)

                return converted_data

            except Exception as e:
                self.logger.error(f"Failed to get real data from {self.current_exchange}, using mock: {e}")

        # Fallback to mock data
        return self._get_mock_data(symbol)
        
    def _get_mock_data(self, symbol: str = 'BTC/USD') -> Dict:
        """Generate mock market data"""
        
        # Update current price for this symbol
        if symbol in self.current_prices:
            volatility = self.volatility_config.get(symbol, 0.02)
            change = random.uniform(-volatility, volatility)
            trend_change = self.trend * random.uniform(0.5, 1.5)
            
            self.current_prices[symbol] *= (1 + change + trend_change)
            
            # Add some realistic bounds
            base_price = 45000 if 'BTC' in symbol else self.current_prices[symbol]
            if self.current_prices[symbol] < base_price * 0.8:
                self.current_prices[symbol] = base_price * 0.8
            elif self.current_prices[symbol] > base_price * 1.2:
                self.current_prices[symbol] = base_price * 1.2
        
        current_price = self.current_prices.get(symbol, 45000)
        
        # Generate realistic bid/ask spread
        spread = current_price * 0.0001  # 0.01% spread
        bid = current_price - spread/2
        ask = current_price + spread/2
        
        # Generate volume
        volume = random.uniform(0.1, 5.0)
        
        # Create market data
        market_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price': current_price,
            'bid': bid,
            'ask': ask,
            'volume': volume,
            'change_24h': random.uniform(-5, 5),
            'high_24h': current_price * random.uniform(1.0, 1.02),
            'low_24h': current_price * random.uniform(0.98, 1.0),
            'market_cap': current_price * 19000000,  # Approximate BTC supply
            'volatility': volatility,
            'source': 'mock_fallback'
        }
        
        self._update_price_history(market_data, symbol)
        return market_data
        
    def _update_price_history(self, market_data: Dict, symbol: str):
        """Update price history with new data"""
        
        # Ensure symbol exists in price history
        if symbol not in self.price_history:
            self.price_history[symbol] = {}
            for timeframe in ["1m", "3m", "15m", "30m", "1h", "4h", "1d"]:
                self.price_history[symbol][timeframe] = []
        
        # Add to 1m history (most recent)
        new_entry = {
            'timestamp': market_data['timestamp'],
            'price': market_data['price'],
            'open': market_data['price'] * (1 + random.uniform(-0.001, 0.001)),
            'high': market_data.get('high_24h', market_data['price'] * 1.002),
            'low': market_data.get('low_24h', market_data['price'] * 0.998),
            'close': market_data['price'],
            'volume': market_data['volume']
        }
        
        self.price_history[symbol]["1m"].append(new_entry)
        
        # Keep only last 200 records
        if len(self.price_history[symbol]["1m"]) > 200:
            self.price_history[symbol]["1m"] = self.price_history[symbol]["1m"][-200:]
            
        self.volume_history.append(market_data['volume'])
        if len(self.volume_history) > 200:
            self.volume_history = self.volume_history[-200:]
        
    def get_historical_data(self, symbol: str = 'BTC/USD', timeframe: str = '1m', limit: int = 100) -> List[Dict]:
        """Get historical market data"""
        if symbol in self.price_history and timeframe in self.price_history[symbol]:
            return self.price_history[symbol][timeframe][-limit:]
        return []
        
    def get_order_book(self, symbol: str = 'BTC/USD', depth: int = 10) -> Dict:
        """Get order book data"""
        # For now, simulate order book (Binance order book API could be added later)
        current_data = self.get_current_data(symbol)
        current_price = current_data['price']
        
        bids = []
        asks = []
        
        for i in range(depth):
            # Generate bids (decreasing prices)
            bid_price = current_price * (1 - (i + 1) * 0.0001)
            bid_quantity = random.uniform(0.1, 2.0)
            bids.append([bid_price, bid_quantity])
            
            # Generate asks (increasing prices)
            ask_price = current_price * (1 + (i + 1) * 0.0001)
            ask_quantity = random.uniform(0.1, 2.0)
            asks.append([ask_price, ask_quantity])
            
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }
        
    def execute_trade(self, action: str, quantity: float, symbol: str = 'BTC/USD', price: float = None) -> Dict:
        """Execute trade via current exchange API or simulate"""

        # Get current exchange connector
        connector = self._get_exchange_connector()
        if connector and connector.is_connected():
            try:
                # Convert symbol to exchange format
                exchange_symbol = self.symbol_maps.get(self.current_exchange, {}).get(symbol, symbol)

                result = connector.execute_trade(action, exchange_symbol, quantity, price)

                if result.get('success'):
                    self.logger.info(f"✓ Real trade executed on {self.current_exchange}: {action} {quantity} {symbol}")
                    return {
                        'success': True,
                        'action': action,
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': result['price'],
                        'timestamp': datetime.now().isoformat(),
                        'fees': result.get('fees', 0),
                        'order_id': result['order_id'],
                        'source': f'{self.current_exchange}_live',
                        'exchange': self.current_exchange
                    }
                else:
                    self.logger.error(f"Trade failed on {self.current_exchange}: {result.get('error')}")
                    return result

            except Exception as e:
                self.logger.error(f"Trade execution error on {self.current_exchange}: {e}")

        # Fallback to simulation
        return self._simulate_trade_execution(action, quantity, symbol, price)
        
    def _simulate_trade_execution(self, action: str, quantity: float, symbol: str = 'BTC/USD', price: float = None) -> Dict:
        """Simulate trade execution with realistic slippage"""
        
        # Get current market data
        market_data = self.get_current_data(symbol)
        
        # Use provided price or market price
        if price is None:
            if action.lower() == 'buy':
                execution_price = market_data['ask']
            else:
                execution_price = market_data['bid']
        else:
            execution_price = price
            
        # Simulate slippage based on quantity
        slippage_factor = min(quantity * 0.001, 0.005)  # Max 0.5% slippage
        
        if action.lower() == 'buy':
            execution_price *= (1 + slippage_factor)
        else:
            execution_price *= (1 - slippage_factor)
            
        # Simulate execution delay
        time.sleep(0.1)
        
        return {
            'success': True,
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': execution_price,
            'timestamp': datetime.now().isoformat(),
            'fees': execution_price * quantity * 0.001,  # 0.1% fee
            'slippage': slippage_factor,
            'order_id': f"MOCK_{int(time.time() * 1000)}",
            'source': 'mock_simulation'
        }
        
    def get_account_balance(self, currency: str = 'USDT') -> float:
        """Get account balance from current exchange"""

        # Get current exchange connector
        connector = self._get_exchange_connector()
        if connector and connector.is_connected():
            try:
                return connector.get_account_balance(currency)
            except Exception as e:
                self.logger.error(f"Failed to get real balance from {self.current_exchange}: {e}")

        # Fallback to mock balance
        balances = {
            'USDT': 10000.0,
            'USD': 10000.0,
            'BTC': 0.1,
            'ETH': 2.0
        }
        return balances.get(currency, 0.0)
        
    def start_stream(self, callback=None):
        """Start real-time data streaming"""
        self.running = True
        self.logger.info("Data stream started")
        
        def stream_worker():
            while self.running:
                try:
                    data = self.get_current_data()
                    if callback:
                        callback(data)
                    time.sleep(1)  # Update every second
                except Exception as e:
                    self.logger.error(f"Error in data stream: {e}")
                    
        thread = threading.Thread(target=stream_worker)
        thread.daemon = True
        thread.start()
        
    def stop_stream(self):
        """Stop data streaming"""
        self.running = False
        self.logger.info("Data stream stopped")
        
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return list(self.symbol_maps.get(self.current_exchange, {}).keys())
        
    def get_market_status(self) -> Dict:
        """Get market status information"""
        exchange_status = {}
        for exchange_name, connector in self.exchanges.items():
            exchange_status[f'{exchange_name}_connected'] = connector.is_connected()

        return {
            'status': 'open',
            'next_open': None,
            'next_close': None,
            'timezone': 'UTC',
            'current_exchange': self.current_exchange,
            'available_exchanges': self.get_available_exchanges(),
            **exchange_status
        }

    def get_connection_info(self) -> Dict:
        """Get detailed connection information for all exchanges"""
        connection_info = {
            'current_exchange': self.current_exchange,
            'available_exchanges': self.get_available_exchanges(),
            'exchanges': {}
        }

        for exchange_name, connector in self.exchanges.items():
            connection_info['exchanges'][exchange_name] = connector.get_connection_status()

        return connection_info

    # Keep original method names for compatibility
    def simulate_trade_execution(self, action: str, quantity: float, symbol: str = 'BTC/USD') -> Dict:
        """Legacy method for compatibility"""
        return self.execute_trade(action, quantity, symbol)