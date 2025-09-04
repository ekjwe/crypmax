"""
Mock data feed for QuantumTrade Bot
Simulates real-time cryptocurrency market data
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class DataFeed:
    """Mock data feed that simulates real cryptocurrency market data"""
    
    def __init__(self, config):
        self.config = config
        self.current_prices = {
            "BTC/USD": 45000.0,
            "ETH/USD": 3200.0,
            "BNB/USD": 320.0,
            "DOGE/USD": 0.12
        }
        self.price_history = {}
        self.last_update = None
        self.volatility = {
            "BTC/USD": 0.02,
            "ETH/USD": 0.025,
            "BNB/USD": 0.03,
            "DOGE/USD": 0.05
        }
        
        # Initialize price history
        self._initialize_price_history()
        
        logger.info("Mock data feed initialized")
    
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
        volatility = self.volatility[pair]
        
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
            
            # Generate volume (higher volume on larger price moves)
            price_change = abs(close_price - open_price) / open_price
            base_volume = random.uniform(100, 1000)
            volume = base_volume * (1 + price_change * 10)
            
            candle = {
                "timestamp": timestamp.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2)
            }
            
            data.append(candle)
            base_price = close_price  # Next candle starts from this close
        
        return data
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return timeframe_map.get(timeframe, 1)
    
    async def get_ohlcv(self, pair: str, timeframe: str, limit: int = 100) -> List[Dict]:
        """Get OHLCV data for a trading pair"""
        try:
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            # Update current prices if needed
            await self._update_prices()
            
            # Get historical data
            if pair not in self.price_history:
                logger.warning(f"Pair {pair} not found in price history")
                return []
            
            if timeframe not in self.price_history[pair]:
                logger.warning(f"Timeframe {timeframe} not found for {pair}")
                return []
            
            data = self.price_history[pair][timeframe]
            
            # Add new candle if enough time has passed
            self._update_current_candle(pair, timeframe)
            
            # Return requested number of candles
            return data[-limit:] if data else []
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            return []
    
    async def get_ticker(self, pair: str) -> Optional[Dict]:
        """Get current ticker information"""
        try:
            await asyncio.sleep(0.05)
            await self._update_prices()
            
            if pair not in self.current_prices:
                return None
            
            current_price = self.current_prices[pair]
            
            # Get 24h data
            daily_data = await self.get_ohlcv(pair, "1h", 24)
            
            if daily_data:
                price_24h_ago = daily_data[0]["close"]
                change_24h = current_price - price_24h_ago
                change_24h_percent = (change_24h / price_24h_ago) * 100
                
                high_24h = max(candle["high"] for candle in daily_data)
                low_24h = min(candle["low"] for candle in daily_data)
                volume_24h = sum(candle["volume"] for candle in daily_data)
            else:
                change_24h = 0
                change_24h_percent = 0
                high_24h = current_price
                low_24h = current_price
                volume_24h = 1000
            
            return {
                "symbol": pair,
                "price": current_price,
                "change_24h": round(change_24h, 2),
                "change_24h_percent": round(change_24h_percent, 2),
                "high_24h": round(high_24h, 2),
                "low_24h": round(low_24h, 2),
                "volume_24h": round(volume_24h, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker: {e}")
            return None
    
    async def _update_prices(self):
        """Update current prices with realistic movements"""
        current_time = time.time()
        
        # Update prices every 30 seconds
        if self.last_update and current_time - self.last_update < 30:
            return
        
        for pair in self.current_prices:
            volatility = self.volatility[pair]
            
            # Generate price movement
            change_percent = random.gauss(0, volatility)
            new_price = self.current_prices[pair] * (1 + change_percent)
            
            # Ensure price doesn't go negative or too extreme
            min_price = self.current_prices[pair] * 0.5
            max_price = self.current_prices[pair] * 2.0
            
            self.current_prices[pair] = max(min_price, min(max_price, new_price))
        
        self.last_update = current_time
    
    def _update_current_candle(self, pair: str, timeframe: str):
        """Update the current candle with latest price"""
        try:
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            current_time = datetime.now()
            
            # Check if we need a new candle
            if not self.price_history[pair][timeframe]:
                return
            
            last_candle = self.price_history[pair][timeframe][-1]
            last_time = datetime.fromisoformat(last_candle["timestamp"])
            
            time_diff = (current_time - last_time).total_seconds() / 60
            
            if time_diff >= timeframe_minutes:
                # Create new candle
                current_price = self.current_prices[pair]
                
                # Generate realistic OHLC for the new period
                open_price = last_candle["close"]
                volatility = self.volatility[pair]
                
                # Generate some movement during the period
                price_moves = []
                for _ in range(5):  # 5 price points during the candle
                    change = random.gauss(0, volatility * 0.3)
                    price_moves.append(open_price * (1 + change))
                
                price_moves.append(current_price)  # Include current price
                
                high_price = max(max(price_moves), open_price)
                low_price = min(min(price_moves), open_price)
                close_price = current_price
                
                # Generate volume
                price_change = abs(close_price - open_price) / open_price
                base_volume = random.uniform(100, 1000)
                volume = base_volume * (1 + price_change * 10)
                
                new_candle = {
                    "timestamp": current_time.isoformat(),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 2)
                }
                
                # Add new candle and maintain history size
                self.price_history[pair][timeframe].append(new_candle)
                if len(self.price_history[pair][timeframe]) > 500:
                    self.price_history[pair][timeframe] = self.price_history[pair][timeframe][-400:]
            
        except Exception as e:
            logger.error(f"Error updating current candle: {e}")
    
    async def get_order_book(self, pair: str, limit: int = 20) -> Dict:
        """Get order book data (mock)"""
        try:
            await asyncio.sleep(0.1)
            
            if pair not in self.current_prices:
                return {"bids": [], "asks": []}
            
            current_price = self.current_prices[pair]
            
            # Generate mock order book
            bids = []
            asks = []
            
            # Generate bids (buy orders) below current price
            for i in range(limit):
                price = current_price * (1 - (i + 1) * 0.001)
                quantity = random.uniform(0.1, 10.0)
                bids.append([round(price, 2), round(quantity, 4)])
            
            # Generate asks (sell orders) above current price
            for i in range(limit):
                price = current_price * (1 + (i + 1) * 0.001)
                quantity = random.uniform(0.1, 10.0)
                asks.append([round(price, 2), round(quantity, 4)])
            
            return {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return {"bids": [], "asks": []}
    
    async def get_trades(self, pair: str, limit: int = 50) -> List[Dict]:
        """Get recent trades data (mock)"""
        try:
            await asyncio.sleep(0.1)
            
            if pair not in self.current_prices:
                return []
            
            current_price = self.current_prices[pair]
            trades = []
            
            # Generate mock recent trades
            for i in range(limit):
                timestamp = datetime.now() - timedelta(seconds=i * 10)
                price = current_price * (1 + random.gauss(0, 0.001))
                quantity = random.uniform(0.01, 5.0)
                side = random.choice(["buy", "sell"])
                
                trades.append({
                    "id": f"trade_{int(time.time())}_{i}",
                    "timestamp": timestamp.isoformat(),
                    "price": round(price, 2),
                    "quantity": round(quantity, 4),
                    "side": side
                })
            
            return sorted(trades, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def get_supported_pairs(self) -> List[str]:
        """Get list of supported trading pairs"""
        return list(self.current_prices.keys())
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a pair"""
        return self.current_prices.get(pair)
    
    async def check_connection(self) -> bool:
        """Check if data feed is connected"""
        await asyncio.sleep(0.01)
        return True
    
    def get_market_status(self) -> Dict:
        """Get market status information"""
        return {
            "status": "open",
            "pairs_count": len(self.current_prices),
            "last_update": self.last_update,
            "uptime": time.time() - (self.last_update or time.time())
        }
