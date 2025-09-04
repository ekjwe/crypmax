"""
Technical Analysis module for QuantumTrade Bot
Implements various technical indicators and pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Technical analysis engine with multiple indicators"""
    
    def __init__(self, config):
        self.config = config
        self.candlestick_patterns = self._get_candlestick_patterns()
    
    def calculate_ema(self, data: List[Dict]) -> Dict:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < 2:
                return {"values": [], "signal": None}
            
            closes = [candle["close"] for candle in data]
            ema_periods = self.config.get("technical_analysis", "ema_periods")
            
            emas = {}
            for period in ema_periods:
                if len(closes) >= period:
                    ema = self._calculate_ema_values(closes, period)
                    emas[f"ema_{period}"] = ema[-1] if ema else None
            
            # Generate signal based on EMA crossover
            signal = self._ema_crossover_signal(emas)
            
            return {
                "values": emas,
                "signal": signal,
                "trend": self._determine_ema_trend(emas)
            }
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return {"values": [], "signal": None}
    
    def _calculate_ema_values(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA values for given period"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]  # First EMA is SMA
        
        for price in prices[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
        
        return ema
    
    def _ema_crossover_signal(self, emas: Dict) -> Optional[str]:
        """Detect EMA crossover signals"""
        if "ema_9" not in emas or "ema_21" not in emas:
            return None
        
        # Simple crossover logic (in real implementation, would check previous values)
        if emas["ema_9"] > emas["ema_21"]:
            return "buy"
        elif emas["ema_9"] < emas["ema_21"]:
            return "sell"
        
        return None
    
    def _determine_ema_trend(self, emas: Dict) -> str:
        """Determine trend based on EMA values"""
        if not emas:
            return "neutral"
        
        # Check if shorter EMAs are above longer EMAs
        ema_values = sorted([(int(k.split("_")[1]), v) for k, v in emas.items() if v is not None])
        
        if len(ema_values) < 2:
            return "neutral"
        
        ascending = all(ema_values[i][1] >= ema_values[i+1][1] for i in range(len(ema_values)-1))
        descending = all(ema_values[i][1] <= ema_values[i+1][1] for i in range(len(ema_values)-1))
        
        if ascending:
            return "bullish"
        elif descending:
            return "bearish"
        else:
            return "neutral"
    
    def calculate_rsi(self, data: List[Dict]) -> Dict:
        """Calculate Relative Strength Index"""
        try:
            if len(data) < 15:
                return {"value": 50, "signal": None}
            
            closes = [candle["close"] for candle in data]
            period = self.config.get("technical_analysis", "rsi_period", 14)
            
            # Calculate price changes
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            
            # Separate gains and losses
            gains = [max(delta, 0) for delta in deltas]
            losses = [abs(min(delta, 0)) for delta in deltas]
            
            # Calculate average gains and losses
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            # Calculate RSI for subsequent periods
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            oversold = self.config.get("technical_analysis", "rsi_oversold", 30)
            overbought = self.config.get("technical_analysis", "rsi_overbought", 70)
            
            signal = None
            if rsi < oversold:
                signal = "buy"
            elif rsi > overbought:
                signal = "sell"
            
            return {
                "value": rsi,
                "signal": signal,
                "extreme": rsi < oversold or rsi > overbought,
                "oversold": rsi < oversold,
                "overbought": rsi > overbought
            }
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return {"value": 50, "signal": None}
    
    def calculate_sma(self, data: List[Dict]) -> Dict:
        """Calculate Simple Moving Average"""
        try:
            if len(data) < 2:
                return {"values": {}, "signal": None}
            
            closes = [candle["close"] for candle in data]
            ma_periods = self.config.get("technical_analysis", "ma_periods", [20, 50, 200])
            
            smas = {}
            for period in ma_periods:
                if len(closes) >= period:
                    sma_values = []
                    for i in range(period - 1, len(closes)):
                        sma = sum(closes[i-period+1:i+1]) / period
                        sma_values.append(sma)
                    smas[f"sma_{period}"] = sma_values[-1] if sma_values else None
            
            # Generate signal based on price vs SMA
            signal = self._sma_signal(closes[-1], smas) if closes else None
            
            return {
                "values": smas,
                "signal": signal,
                "trend": self._determine_sma_trend(closes[-1], smas) if closes else "neutral"
            }
            
        except Exception as e:
            logger.error(f"SMA calculation error: {e}")
            return {"values": {}, "signal": None}
    
    def _sma_signal(self, current_price: float, smas: Dict) -> Optional[str]:
        """Generate signal based on price vs SMA"""
        if "sma_20" not in smas or smas["sma_20"] is None:
            return None
        
        if current_price > smas["sma_20"] * 1.01:  # 1% above SMA
            return "buy"
        elif current_price < smas["sma_20"] * 0.99:  # 1% below SMA
            return "sell"
        
        return None
    
    def _determine_sma_trend(self, current_price: float, smas: Dict) -> str:
        """Determine trend based on SMA alignment"""
        if not smas:
            return "neutral"
        
        sma_values = [v for v in smas.values() if v is not None]
        if not sma_values:
            return "neutral"
        
        above_sma = sum(1 for sma in sma_values if current_price > sma)
        below_sma = len(sma_values) - above_sma
        
        if above_sma > below_sma:
            return "bullish"
        elif below_sma > above_sma:
            return "bearish"
        else:
            return "neutral"
    
    def find_support_resistance(self, data: List[Dict]) -> Dict:
        """Find support and resistance levels"""
        try:
            if len(data) < 10:
                return {"support": None, "resistance": None}
            
            highs = [candle["high"] for candle in data]
            lows = [candle["low"] for candle in data]
            
            # Find local peaks and troughs
            support_levels = self._find_local_extrema(lows, "min")
            resistance_levels = self._find_local_extrema(highs, "max")
            
            # Get the most recent significant levels
            current_price = data[-1]["close"]
            
            support = max([s for s in support_levels if s < current_price], default=None)
            resistance = min([r for r in resistance_levels if r > current_price], default=None)
            
            return {
                "support": support,
                "resistance": resistance,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Support/Resistance calculation error: {e}")
            return {"support": None, "resistance": None}
    
    def _find_local_extrema(self, values: List[float], extrema_type: str) -> List[float]:
        """Find local maxima or minima"""
        extrema = []
        window = 5
        
        for i in range(window, len(values) - window):
            if extrema_type == "max":
                if values[i] == max(values[i-window:i+window+1]):
                    extrema.append(values[i])
            else:  # min
                if values[i] == min(values[i-window:i+window+1]):
                    extrema.append(values[i])
        
        return extrema
    
    def detect_candlestick_patterns(self, data: List[Dict]) -> List[Dict]:
        """Detect candlestick patterns"""
        try:
            if len(data) < 3:
                return []
            
            patterns = []
            recent_candles = data[-3:]  # Check last 3 candles
            
            # Check for doji
            if self._is_doji(recent_candles[-1]):
                patterns.append({
                    "name": "Doji",
                    "signal": "neutral",
                    "action": "hold",
                    "strength": 0.5
                })
            
            # Check for hammer
            if self._is_hammer(recent_candles[-1]):
                patterns.append({
                    "name": "Hammer",
                    "signal": "bullish",
                    "action": "buy",
                    "strength": 0.7
                })
            
            # Check for shooting star
            if self._is_shooting_star(recent_candles[-1]):
                patterns.append({
                    "name": "Shooting Star",
                    "signal": "bearish",
                    "action": "sell",
                    "strength": 0.7
                })
            
            # Check for engulfing patterns
            if len(recent_candles) >= 2:
                engulfing = self._check_engulfing_pattern(recent_candles[-2], recent_candles[-1])
                if engulfing:
                    patterns.append(engulfing)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Candlestick pattern detection error: {e}")
            return []
    
    def _is_doji(self, candle: Dict) -> bool:
        """Check if candle is a doji"""
        body_size = abs(candle["close"] - candle["open"])
        total_range = candle["high"] - candle["low"]
        
        if total_range == 0:
            return True
        
        return body_size / total_range < 0.1
    
    def _is_hammer(self, candle: Dict) -> bool:
        """Check if candle is a hammer"""
        body_size = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        
        return (lower_wick > body_size * 2 and 
                upper_wick < body_size * 0.5 and
                body_size > 0)
    
    def _is_shooting_star(self, candle: Dict) -> bool:
        """Check if candle is a shooting star"""
        body_size = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])
        
        return (upper_wick > body_size * 2 and 
                lower_wick < body_size * 0.5 and
                body_size > 0)
    
    def _check_engulfing_pattern(self, prev_candle: Dict, curr_candle: Dict) -> Optional[Dict]:
        """Check for bullish or bearish engulfing patterns"""
        prev_body_size = abs(prev_candle["close"] - prev_candle["open"])
        curr_body_size = abs(curr_candle["close"] - curr_candle["open"])
        
        # Bullish engulfing
        if (prev_candle["close"] < prev_candle["open"] and  # Previous red
            curr_candle["close"] > curr_candle["open"] and  # Current green
            curr_candle["open"] < prev_candle["close"] and  # Opens below prev close
            curr_candle["close"] > prev_candle["open"] and  # Closes above prev open
            curr_body_size > prev_body_size):  # Larger body
            
            return {
                "name": "Bullish Engulfing",
                "signal": "bullish",
                "action": "buy",
                "strength": 0.8
            }
        
        # Bearish engulfing
        if (prev_candle["close"] > prev_candle["open"] and  # Previous green
            curr_candle["close"] < curr_candle["open"] and  # Current red
            curr_candle["open"] > prev_candle["close"] and  # Opens above prev close
            curr_candle["close"] < prev_candle["open"] and  # Closes below prev open
            curr_body_size > prev_body_size):  # Larger body
            
            return {
                "name": "Bearish Engulfing",
                "signal": "bearish",
                "action": "sell",
                "strength": 0.8
            }
        
        return None
    
    def analyze_volume(self, data: List[Dict]) -> Dict:
        """Analyze volume patterns"""
        try:
            if len(data) < 10:
                return {"spike": False, "trend": "neutral"}
            
            volumes = [candle["volume"] for candle in data]
            recent_volume = volumes[-1]
            avg_volume = sum(volumes[-10:-1]) / 9  # Average of last 9 volumes
            
            volume_threshold = self.config.get("technical_analysis", "volume_threshold", 1.5)
            spike = recent_volume > avg_volume * volume_threshold
            
            # Analyze volume trend
            recent_avg = sum(volumes[-5:]) / 5
            older_avg = sum(volumes[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return {
                "current": recent_volume,
                "average": avg_volume,
                "spike": spike,
                "trend": trend,
                "ratio": recent_volume / avg_volume if avg_volume > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            return {"spike": False, "trend": "neutral"}
    
    def identify_trend(self, data: List[Dict]) -> str:
        """Identify overall trend"""
        try:
            if len(data) < 20:
                return "neutral"
            
            closes = [candle["close"] for candle in data]
            
            # Use linear regression to identify trend
            x = list(range(len(closes)))
            y = closes
            
            # Calculate slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Determine trend based on slope
            if slope > closes[-1] * 0.001:  # Positive slope > 0.1% of current price
                return "bullish"
            elif slope < -closes[-1] * 0.001:  # Negative slope < -0.1% of current price
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Trend identification error: {e}")
            return "neutral"
    
    def detect_breakout(self, data: List[Dict]) -> Dict:
        """Detect breakout patterns"""
        try:
            if len(data) < 20:
                return {"signal": None, "confirmed": False}
            
            # Get support and resistance levels
            sr_levels = self.find_support_resistance(data)
            current_price = data[-1]["close"]
            prev_price = data[-2]["close"]
            
            support = sr_levels.get("support")
            resistance = sr_levels.get("resistance")
            
            # Check for breakouts
            if resistance and current_price > resistance and prev_price <= resistance:
                # Breakout above resistance
                volume_analysis = self.analyze_volume(data)
                confirmed = volume_analysis.get("spike", False)
                
                return {
                    "signal": True,
                    "action": "buy",
                    "type": "resistance_breakout",
                    "level": resistance,
                    "confirmed": confirmed
                }
            
            elif support and current_price < support and prev_price >= support:
                # Breakdown below support
                volume_analysis = self.analyze_volume(data)
                confirmed = volume_analysis.get("spike", False)
                
                return {
                    "signal": True,
                    "action": "sell",
                    "type": "support_breakdown",
                    "level": support,
                    "confirmed": confirmed
                }
            
            return {"signal": None, "confirmed": False}
            
        except Exception as e:
            logger.error(f"Breakout detection error: {e}")
            return {"signal": None, "confirmed": False}
    
    def _get_candlestick_patterns(self) -> List[str]:
        """Get list of supported candlestick patterns"""
        return [
            "Doji", "Hammer", "Shooting Star", "Bullish Engulfing", "Bearish Engulfing",
            "Morning Star", "Evening Star", "Piercing Line", "Dark Cloud Cover",
            "Hanging Man", "Inverted Hammer", "Three White Soldiers", "Three Black Crows",
            "Inside Bar", "Outside Bar", "Pin Bar", "Spinning Top", "Marubozu",
            "Harami", "Tweezer Top", "Tweezer Bottom", "Rising Three Methods",
            "Falling Three Methods", "Three Inside Up", "Three Inside Down",
            "Three Outside Up", "Three Outside Down", "Abandoned Baby", "Belt Hold"
        ]
